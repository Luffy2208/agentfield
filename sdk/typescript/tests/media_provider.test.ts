import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MediaRouter } from '../src/ai/MediaProvider.js';
import type { MediaProvider, MediaResponse } from '../src/ai/MediaProvider.js';
import { OpenRouterMediaProvider } from '../src/ai/OpenRouterMediaProvider.js';

// ── MediaRouter tests ────────────────────────────────────────────────

function makeStubProvider(
  name: string,
  modalities: string[] = ['image', 'audio', 'video']
): MediaProvider {
  const empty: MediaResponse = {
    text: '',
    images: [],
    audio: null,
    files: [],
    videos: [],
    rawResponse: null,
  };
  return {
    name,
    supportedModalities: modalities,
    generateImage: vi.fn().mockResolvedValue(empty),
    generateAudio: vi.fn().mockResolvedValue(empty),
    generateVideo: vi.fn().mockResolvedValue(empty),
  };
}

describe('MediaRouter', () => {
  it('resolves provider by prefix', () => {
    const router = new MediaRouter();
    const prov = makeStubProvider('openrouter');
    router.register('openrouter/', prov);
    expect(router.resolve('openrouter/google/veo-3', 'video')).toBe(prov);
  });

  it('longest prefix wins', () => {
    const router = new MediaRouter();
    const generic = makeStubProvider('generic');
    const specific = makeStubProvider('specific');
    router.register('openrouter/', generic);
    router.register('openrouter/google/', specific);

    expect(router.resolve('openrouter/google/veo-3', 'video')).toBe(specific);
    expect(router.resolve('openrouter/openai/dall-e', 'image')).toBe(generic);
  });

  it('throws when no provider matches', () => {
    const router = new MediaRouter();
    expect(() => router.resolve('unknown/model', 'video')).toThrow(
      "No provider for model 'unknown/model' with 'video' capability"
    );
  });

  it('checks capability filter', () => {
    const router = new MediaRouter();
    const imageOnly = makeStubProvider('img', ['image']);
    router.register('img/', imageOnly);

    expect(router.resolve('img/model', 'image')).toBe(imageOnly);
    expect(() => router.resolve('img/model', 'video')).toThrow();
  });
});

// ── OpenRouterMediaProvider tests ────────────────────────────────────

describe('OpenRouterMediaProvider', () => {
  const originalFetch = globalThis.fetch;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockFetch = vi.fn();
    globalThis.fetch = mockFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('throws without API key', () => {
    const orig = process.env.OPENROUTER_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    expect(() => new OpenRouterMediaProvider()).toThrow('API key required');
    if (orig) process.env.OPENROUTER_API_KEY = orig;
  });

  it('accepts apiKey in constructor', () => {
    const p = new OpenRouterMediaProvider({ apiKey: 'test-key' });
    expect(p.name).toBe('openrouter');
    expect(p.supportedModalities).toContain('video');
  });

  it('strips openrouter/ prefix from model', async () => {
    const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

    // Mock image generation
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        choices: [
          {
            message: {
              content: [
                { type: 'text', text: 'Generated' },
                { type: 'image_url', image_url: { url: 'https://example.com/img.png' } },
              ],
            },
          },
        ],
      }),
    });

    await provider.generateImage({ prompt: 'a cat', model: 'openrouter/openai/gpt-image-1' });

    // Verify the model sent in the body has prefix stripped
    const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(callBody.model).toBe('openai/gpt-image-1');
  });

  describe('generateVideo', () => {
    it('submits job, polls, downloads', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });
      const videoBytes = Buffer.from('fake-video-data');

      // Submit
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-123' }),
      });
      // Poll -> completed
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'job-123',
          status: 'completed',
          unsigned_url: 'https://example.com/video.mp4',
          cost_usd: 0.05,
        }),
      });
      // Download
      mockFetch.mockResolvedValueOnce({
        ok: true,
        arrayBuffer: async () => videoBytes.buffer.slice(
          videoBytes.byteOffset,
          videoBytes.byteOffset + videoBytes.byteLength
        ),
      });

      const resp = await provider.generateVideo({
        prompt: 'a sunset',
        model: 'openrouter/google/veo-3',
        pollInterval: 1, // 1ms for test speed
      });

      expect(resp.videos).toHaveLength(1);
      expect(resp.videos[0].url).toBe('https://example.com/video.mp4');
      expect(resp.videos[0].data).toBe(videoBytes.toString('base64'));
      expect(resp.videos[0].costUsd).toBe(0.05);

      // Verify calls: submit, poll, download
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it('throws on submit failure', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 402,
        text: async () => 'Insufficient credits',
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1 })
      ).rejects.toThrow('Video submit failed: 402');
    });

    it('throws on generation failure status', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-fail' }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-fail', status: 'failed', error: 'content policy' }),
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1 })
      ).rejects.toThrow('Video generation failed');
    });

    it('throws on timeout', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 'job-slow' }),
      });
      // Always return pending
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({ id: 'job-slow', status: 'pending' }),
      });

      await expect(
        provider.generateVideo({ prompt: 'test', pollInterval: 1, timeout: 10 })
      ).rejects.toThrow('timed out');
    });
  });

  describe('generateImage', () => {
    it('extracts images from response', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: [
                  { type: 'text', text: 'Here is your image' },
                  { type: 'image_url', image_url: { url: 'https://cdn.example.com/img.png' } },
                ],
              },
            },
          ],
        }),
      });

      const resp = await provider.generateImage({ prompt: 'a cat' });
      expect(resp.text).toBe('Here is your image');
      expect(resp.images).toHaveLength(1);
      expect(resp.images[0].url).toBe('https://cdn.example.com/img.png');
    });

    it('handles data URL images with b64', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: [
                  {
                    type: 'image_url',
                    image_url: { url: 'data:image/png;base64,iVBOR' },
                  },
                ],
              },
            },
          ],
        }),
      });

      const resp = await provider.generateImage({ prompt: 'test' });
      expect(resp.images[0].b64Json).toBe('iVBOR');
    });
  });

  describe('generateAudio', () => {
    it('parses SSE stream and collects audio chunks', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      const sseLines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        'data: {"choices":[{"delta":{"audio":{"data":"AAAA"}}}]}\n\n',
        'data: {"choices":[{"delta":{"audio":{"data":"BBBB"}}}]}\n\n',
        'data: [DONE]\n\n',
      ];
      const encoder = new TextEncoder();
      let callIndex = 0;

      const mockReader = {
        read: vi.fn().mockImplementation(async () => {
          if (callIndex < sseLines.length) {
            const chunk = encoder.encode(sseLines[callIndex]);
            callIndex++;
            return { done: false, value: chunk };
          }
          return { done: true, value: undefined };
        }),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: { getReader: () => mockReader },
      });

      const resp = await provider.generateAudio({ text: 'say hello' });
      expect(resp.text).toBe('Hello');
      expect(resp.audio).not.toBeNull();
      expect(resp.audio!.data).toBe('AAAABBBB');
      expect(resp.audio!.format).toBe('wav');
    });

    it('throws on failure', async () => {
      const provider = new OpenRouterMediaProvider({ apiKey: 'test-key' });

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Internal error',
      });

      await expect(
        provider.generateAudio({ text: 'test' })
      ).rejects.toThrow('Audio generation failed: 500');
    });
  });
});
