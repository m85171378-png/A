import { GoogleGenAI, Modality } from '@google/genai';

// --- Utilities ---

/**
 * Converts Float32 audio samples to Int16 PCM.
 */
function floatTo16BitPCM(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
}

/**
 * Converts Int16 PCM to Float32 audio samples.
 */
function int16ToFloat32(input: Int16Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = input[i] / 32768;
  }
  return output;
}

/**
 * Base64 helper
 */
const base64ToArrayBuffer = (base64: string) => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};

const arrayBufferToBase64 = (buffer: ArrayBuffer) => {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

// --- App Logic ---

class EthioAIApp {
  private ai: GoogleGenAI;
  private session: any = null;
  private audioContext: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private isConnected = false;
  private isMicMuted = false;
  private isAudioOutputMuted = false;
  private nextStreamTime = 0;
  private selectedVoice = 'Zephyr';

  // UI Elements
  private statusDot = document.getElementById('status-dot')!;
  private statusText = document.getElementById('status-text')!;
  private aiStatus = document.getElementById('ai-status')!;
  private connectBtn = document.getElementById('connect-btn') as HTMLButtonElement;
  private toggleMicBtn = document.getElementById('toggle-mic')!;
  private toggleAudioBtn = document.getElementById('toggle-audio')!;
  private voiceSelect = document.getElementById('voice-select') as HTMLSelectElement;
  private chatMessages = document.getElementById('chat-messages')!;
  private chatInput = document.getElementById('chat-input') as HTMLInputElement;
  private sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
  private aiOrb = document.getElementById('ai-orb')!;
  private clearChatBtn = document.getElementById('clear-chat')!;

  constructor() {
    this.ai = new GoogleGenAI({
      apiKey: (process.env as any).GEMINI_API_KEY || '',
      apiVersion: 'v1alpha',
    });

    this.setupEventListeners();
    this.addMessage('system', 'Welcome to Ethio AI! Click "Connect" to start a live conversation.');
  }

  private setupEventListeners() {
    this.connectBtn.addEventListener('click', () => this.toggleConnection());
    this.toggleMicBtn.addEventListener('click', () => this.toggleMic());
    this.toggleAudioBtn.addEventListener('click', () => this.toggleAudioOutput());
    this.sendBtn.addEventListener('click', () => this.sendTextMessage());
    this.clearChatBtn.addEventListener('click', () => {
      this.chatMessages.innerHTML = '';
      this.addMessage('system', 'History cleared.');
    });
    this.chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.sendTextMessage();
    });
    this.voiceSelect.addEventListener('change', () => {
      this.selectedVoice = this.voiceSelect.value;
      if (this.isConnected) {
        this.addMessage('system', `Voice changed to ${this.selectedVoice}. Please reconnect for the change to take effect.`);
      }
    });
  }

  private async toggleConnection() {
    if (this.isConnected) {
      this.disconnect();
    } else {
      await this.connect();
    }
  }

  private updateStatusUI(connected: boolean) {
    this.isConnected = connected;
    if (connected) {
      this.statusDot.classList.replace('bg-zinc-600', 'bg-green-500');
      this.statusText.textContent = 'Connected';
      this.statusText.classList.replace('text-zinc-400', 'text-green-400');
      this.connectBtn.textContent = 'Disconnect';
      this.connectBtn.classList.add('bg-red-500', 'text-white');
      this.connectBtn.classList.remove('bg-white', 'text-zinc-950');
      this.aiStatus.textContent = 'Ethio AI is listening...';
    } else {
      this.statusDot.classList.replace('bg-green-500', 'bg-zinc-600');
      this.statusText.textContent = 'Disconnected';
      this.statusText.classList.replace('text-green-400', 'text-zinc-400');
      this.connectBtn.textContent = 'Connect';
      this.connectBtn.classList.remove('bg-red-500', 'text-white');
      this.connectBtn.classList.add('bg-white', 'text-zinc-950');
      this.aiStatus.textContent = 'Connect to start talking';
    }
  }

  private async connect() {
    try {
      this.addMessage('system', 'Connecting to Ethio AI...');
      
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      this.nextStreamTime = this.audioContext.currentTime;

      this.session = await this.ai.live.connect({
        model: 'gemini-3.1-flash-live-preview',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: this.selectedVoice } },
          },
          systemInstruction: "You are Ethio AI, a friendly and knowledgeable AI assistant specializing in everything about Ethiopia—its culture, history, geography, food, and people. You respond warmly and concisely. You can hear and speak in real-time.",
        },
        callbacks: {
          onopen: () => {
            this.updateStatusUI(true);
            this.startMic();
            this.addMessage('system', 'Connection established.');
          },
          onmessage: (message: any) => {
            this.handleServerMessage(message);
          },
          onclose: () => {
            this.disconnect();
            this.addMessage('system', 'Connection closed.');
          },
          onerror: (err: any) => {
            console.error('Session Error:', err);
            this.addMessage('system', `Error: ${err.message || 'Unknown error'}`);
            this.disconnect();
          },
        },
      });
    } catch (error: any) {
      console.error('Connection failed:', error);
      this.addMessage('system', `Failed to connect: ${error.message}`);
    }
  }

  private disconnect() {
    if (this.session) {
      this.session.close();
      this.session = null;
    }
    this.stopMic();
    this.updateStatusUI(false);
  }

  private async startMic() {
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = this.audioContext!.createMediaStreamSource(this.micStream);
      this.processor = this.audioContext!.createScriptProcessor(2048, 1, 1);

      source.connect(this.processor);
      this.processor.connect(this.audioContext!.destination);

      this.processor.onaudioprocess = (e) => {
        if (this.isMicMuted || !this.isConnected || !this.session) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = floatTo16BitPCM(inputData);
        const base64 = arrayBufferToBase64(pcm16.buffer);

        this.session.sendRealtimeInput({
          audio: { data: base64, mimeType: 'audio/pcm;rate=16000' }
        });
        
        // Visual feedback
        const volume = inputData.reduce((acc, val) => acc + Math.abs(val), 0) / inputData.length;
        const scale = 1 + volume * 5;
        this.aiOrb.style.transform = `scale(${scale})`;
      };
    } catch (err) {
      console.error('Microphone access denied:', err);
      this.addMessage('system', 'Microphone access denied. You can still use text chat.');
    }
  }

  private stopMic() {
    if (this.micStream) {
      this.micStream.getTracks().forEach(t => t.stop());
      this.micStream = null;
    }
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    this.aiOrb.style.transform = 'scale(1)';
  }

  private toggleMic() {
    this.isMicMuted = !this.isMicMuted;
    const icon = this.toggleMicBtn.querySelector('svg');
    if (icon) {
      (icon as any).style.stroke = this.isMicMuted ? '#ef4444' : '#a1a1aa';
    }
    this.addMessage('system', this.isMicMuted ? 'Microphone muted' : 'Microphone active');
  }

  private toggleAudioOutput() {
    this.isAudioOutputMuted = !this.isAudioOutputMuted;
    const icon = this.toggleAudioBtn.querySelector('svg');
    if (icon) {
      (icon as any).style.stroke = this.isAudioOutputMuted ? '#ef4444' : '#a1a1aa';
    }
    this.addMessage('system', this.isAudioOutputMuted ? 'AI voice silenced' : 'AI voice active');
  }

  private handleServerMessage(message: any) {
    // 1. Handle Audio Output
    const parts = message.serverContent?.modelTurn?.parts;
    if (parts) {
      for (const part of parts) {
        if (part.inlineData?.data && !this.isAudioOutputMuted) {
          this.playAudioChunk(part.inlineData.data);
        }
        if (part.text) {
          this.addMessage('ai', part.text);
        }
      }
    }

    // 2. Handle Interruption
    if (message.serverContent?.interrupted) {
      this.stopPlayback();
    }
  }

  private playAudioChunk(base64: string) {
    if (!this.audioContext) return;

    const buffer = base64ToArrayBuffer(base64);
    const pcm16 = new Int16Array(buffer);
    const float32 = int16ToFloat32(pcm16);

    const audioBuffer = this.audioContext.createBuffer(1, float32.length, 16000);
    audioBuffer.getChannelData(0).set(float32);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const startTime = Math.max(this.audioContext.currentTime, this.nextStreamTime);
    source.start(startTime);
    this.nextStreamTime = startTime + audioBuffer.duration;
  }

  private stopPlayback() {
    this.nextStreamTime = this.audioContext?.currentTime || 0;
    this.aiStatus.textContent = 'Interrupted...';
    setTimeout(() => {
      if (this.isConnected) this.aiStatus.textContent = 'Ethio AI is listening...';
    }, 1000);
  }

  private async sendTextMessage() {
    const text = this.chatInput.value.trim();
    if (!text) return;

    this.chatInput.value = '';
    this.addMessage('user', text);

    if (this.isConnected && this.session) {
      this.session.sendRealtimeInput({ text });
    } else {
      this.addMessage('system', 'Not connected. Connect to talk to Ethio AI.');
    }
  }

  private addMessage(sender: 'user' | 'ai' | 'system', text: string) {
    if (!text) return;
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `flex flex-col ${sender === 'user' ? 'items-end' : 'items-start animate-in fade-in slide-in-from-bottom-2'}`;
    
    let senderName = 'Ethio AI';
    let colorClass = 'bg-zinc-800 text-zinc-100';
    if (sender === 'user') {
      senderName = 'You';
      colorClass = 'bg-purple-600 text-white';
    } else if (sender === 'system') {
      senderName = 'System';
      colorClass = 'bg-zinc-900/50 text-zinc-500 text-[10px] uppercase font-bold italic border border-zinc-800/50';
    }

    msgDiv.innerHTML = `
      <span class="text-[10px] font-bold uppercase tracking-wider mb-1 px-1 ${sender === 'user' ? 'text-zinc-500' : 'text-zinc-400'}">${senderName}</span>
      <div class="px-4 py-2 rounded-2xl max-w-[85%] text-sm shadow-sm ${colorClass} whitespace-pre-wrap">
        ${text}
      </div>
    `;

    this.chatMessages.appendChild(msgDiv);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }
}

// Initialize the app
window.addEventListener('DOMContentLoaded', () => {
  new EthioAIApp();
});
