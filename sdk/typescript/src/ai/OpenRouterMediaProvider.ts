/**
 * OpenRouter-backed MediaProvider implementation.
 * Supports video generation (async job), image generation, and audio generation via SSE.
 */

import type {
  MediaProvider,
  MediaResponse,
  VideoRequest,
  ImageRequest,
  AudioRequest,
} from './MediaProvider.js';

const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';

const DEFAULT_POLL_INTERVAL = 30_000; // 30s
const DEFAULT_TIMEOUT = 600_000; // 10min

function emptyMediaResponse(raw: unknown): MediaResponse {
  return { text: '', images: [], audio: null, files: [], videos: [], rawResponse: raw };
}

function stripPrefix(model: string): string {
  return model.startsWith('openrouter/') ? model.slice('openrouter/'.length) : model;
}

export interface OpenRouterMediaProviderOptions {
  apiKey?: string;
  baseUrl?: string;
}

export class OpenRouterMediaProvider implements MediaProvider {
  readonly name = 'openrouter';
  readonly supportedModalities = ['image', 'audio', 'video'];

  private readonly apiKey: string;
  private readonly baseUrl: string;

  constructor(options: OpenRouterMediaProviderOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.OPENROUTER_API_KEY ?? '';
    this.baseUrl = options.baseUrl ?? OPENROUTER_BASE;
    if (!this.apiKey) {
      throw new Error('OpenRouter API key required: pass apiKey or set OPENROUTER_API_KEY');
    }
  }

  // ── Video ──────────────────────────────────────────────────────────

  async generateVideo(request: VideoRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? 'google/veo-3');
    const pollInterval = request.pollInterval ?? DEFAULT_POLL_INTERVAL;
    const timeout = request.timeout ?? DEFAULT_TIMEOUT;

    // Build request body
    const body: Record<string, unknown> = {
      model,
      prompt: request.prompt,
    };
    if (request.duration != null) body.duration = request.duration;
    if (request.resolution) body.resolution = request.resolution;
    if (request.aspectRatio) body.aspect_ratio = request.aspectRatio;
    if (request.generateAudio != null) body.generate_audio = request.generateAudio;
    if (request.seed != null) body.seed = request.seed;
    if (request.frameImages) body.frame_images = request.frameImages;
    if (request.inputReferences) body.input_references = request.inputReferences;

    // Submit job
    const submitRes = await this.post(`${this.baseUrl}/videos`, body);
    if (!submitRes.ok) {
      throw new Error(`Video submit failed: ${submitRes.status} ${await submitRes.text()}`);
    }
    const submitData = (await submitRes.json()) as Record<string, unknown>;
    const jobId = submitData.id as string;
    if (!jobId) {
      throw new Error('No job id returned from video submit');
    }

    // Poll until done
    const deadline = Date.now() + timeout;
    let jobData: Record<string, unknown> = {};
    while (Date.now() < deadline) {
      await sleep(pollInterval);
      const pollRes = await this.get(`${this.baseUrl}/videos/${jobId}`);
      if (!pollRes.ok) {
        throw new Error(`Video poll failed: ${pollRes.status} ${await pollRes.text()}`);
      }
      jobData = (await pollRes.json()) as Record<string, unknown>;
      const status = jobData.status as string | undefined;
      if (status === 'completed') break;
      if (status === 'failed' || status === 'error') {
        throw new Error(`Video generation failed: ${JSON.stringify(jobData)}`);
      }
    }

    if ((jobData.status as string) !== 'completed') {
      throw new Error('Video generation timed out');
    }

    // Extract video URL
    const unsignedUrl = jobData.unsigned_url as string | undefined;
    const signedUrl = jobData.url as string | undefined;
    const videoUrl = unsignedUrl ?? signedUrl;

    // Download video bytes if URL available
    let videoData: string | undefined;
    if (videoUrl) {
      const dlRes = await fetch(videoUrl);
      if (dlRes.ok) {
        const buf = Buffer.from(await dlRes.arrayBuffer());
        videoData = buf.toString('base64');
      }
    }

    const resp = emptyMediaResponse(jobData);
    resp.videos.push({
      url: videoUrl,
      data: videoData,
      mimeType: 'video/mp4',
      duration: request.duration,
      resolution: request.resolution,
      aspectRatio: request.aspectRatio,
      hasAudio: request.generateAudio,
      costUsd: jobData.cost_usd as number | undefined,
    });
    return resp;
  }

  // ── Image ──────────────────────────────────────────────────────────

  async generateImage(request: ImageRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? 'openai/gpt-image-1');

    const messages: unknown[] = [{ role: 'user', content: request.prompt }];
    const body: Record<string, unknown> = {
      model,
      messages,
      modalities: ['image', 'text'],
    };
    if (request.size) body.size = request.size;
    if (request.quality) body.quality = request.quality;
    if (request.imageConfig) body.image_config = request.imageConfig;

    const res = await this.post(`${this.baseUrl}/chat/completions`, body);
    if (!res.ok) {
      throw new Error(`Image generation failed: ${res.status} ${await res.text()}`);
    }
    const data = (await res.json()) as Record<string, unknown>;
    const resp = emptyMediaResponse(data);

    // Extract images from choices
    const choices = data.choices as Array<Record<string, unknown>> | undefined;
    if (choices) {
      for (const choice of choices) {
        const msg = choice.message as Record<string, unknown> | undefined;
        if (!msg) continue;
        // Text
        if (typeof msg.content === 'string') {
          resp.text += msg.content;
        }
        // Content array (multimodal)
        if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            const p = part as Record<string, unknown>;
            if (p.type === 'text') {
              resp.text += p.text as string;
            } else if (p.type === 'image_url') {
              const imgUrl = p.image_url as Record<string, unknown> | undefined;
              const url = imgUrl?.url as string | undefined;
              if (url?.startsWith('data:')) {
                const b64 = url.split(',', 2)[1];
                resp.images.push({ url, b64Json: b64 });
              } else if (url) {
                resp.images.push({ url });
              }
            }
          }
        }
      }
    }

    return resp;
  }

  // ── Audio ──────────────────────────────────────────────────────────

  async generateAudio(request: AudioRequest): Promise<MediaResponse> {
    const model = stripPrefix(request.model ?? 'openai/gpt-4o-mini-tts');

    const messages: unknown[] = [{ role: 'user', content: request.text }];
    const body: Record<string, unknown> = {
      model,
      messages,
      modalities: ['text', 'audio'],
      stream: true,
      audio: {
        voice: request.voice ?? 'alloy',
        format: request.format ?? 'wav',
      },
    };

    const res = await this.post(`${this.baseUrl}/chat/completions`, body);
    if (!res.ok) {
      throw new Error(`Audio generation failed: ${res.status} ${await res.text()}`);
    }

    // Parse SSE stream and collect audio chunks
    const audioChunks: string[] = [];
    let textContent = '';
    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error('No response body stream available');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // Keep last incomplete line in buffer
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') continue;

        try {
          const chunk = JSON.parse(payload) as Record<string, unknown>;
          const choices = chunk.choices as Array<Record<string, unknown>> | undefined;
          if (!choices) continue;
          for (const choice of choices) {
            const delta = choice.delta as Record<string, unknown> | undefined;
            if (!delta) continue;
            if (typeof delta.content === 'string') {
              textContent += delta.content;
            }
            const audioDelta = delta.audio as Record<string, unknown> | undefined;
            if (audioDelta?.data) {
              audioChunks.push(audioDelta.data as string);
            }
          }
        } catch {
          // skip malformed chunks
        }
      }
    }

    const resp = emptyMediaResponse(null);
    resp.text = textContent;
    if (audioChunks.length > 0) {
      resp.audio = {
        data: audioChunks.join(''),
        format: request.format ?? 'wav',
      };
    }
    return resp;
  }

  // ── Helpers ────────────────────────────────────────────────────────

  private post(url: string, body: unknown): Promise<Response> {
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });
  }

  private get(url: string): Promise<Response> {
    return fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
      },
    });
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
