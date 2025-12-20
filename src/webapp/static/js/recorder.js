/**
 * Web Audio API-based audio recorder
 * Records audio from microphone and uploads to server as Base64 WAV
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.isRecording = false;
        this.startTime = null;
    }

    /**
     * Initialize and start recording
     */
    async start() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });

            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm',
            });

            this.audioChunks = [];
            this.startTime = Date.now();

            // Collect audio data chunks
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            // Start recording
            this.mediaRecorder.start();
            this.isRecording = true;

            console.log('Recording started');
            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            throw new Error('マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。');
        }
    }

    /**
     * Stop recording and return audio Blob
     */
    async stop() {
        return new Promise((resolve, reject) => {
            if (!this.mediaRecorder || !this.isRecording) {
                reject(new Error('録音が開始されていません'));
                return;
            }

            this.mediaRecorder.onstop = async () => {
                // Calculate duration
                const duration = (Date.now() - this.startTime) / 1000;
                console.log(`Recording stopped. Duration: ${duration.toFixed(2)}s`);

                // Create audio blob from chunks
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

                // Convert to WAV format using Web Audio API
                try {
                    const wavBlob = await this.convertToWav(audioBlob);

                    // Stop all tracks
                    if (this.stream) {
                        this.stream.getTracks().forEach(track => track.stop());
                    }

                    this.isRecording = false;
                    resolve(wavBlob);
                } catch (error) {
                    reject(error);
                }
            };

            this.mediaRecorder.stop();
        });
    }

    /**
     * Convert audio blob to WAV format
     */
    async convertToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();

        try {
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Convert to WAV
            const wavBuffer = this.audioBufferToWav(audioBuffer);
            const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });

            return wavBlob;
        } finally {
            await audioContext.close();
        }
    }

    /**
     * Convert AudioBuffer to WAV format
     * Based on: https://github.com/mattdiamond/Recorderjs
     */
    audioBufferToWav(buffer) {
        const numberOfChannels = 1; // Force mono
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        // Get mono channel data
        let channelData;
        if (buffer.numberOfChannels === 1) {
            channelData = buffer.getChannelData(0);
        } else {
            // Mix down to mono
            const left = buffer.getChannelData(0);
            const right = buffer.getChannelData(1);
            channelData = new Float32Array(left.length);
            for (let i = 0; i < left.length; i++) {
                channelData[i] = (left[i] + right[i]) / 2;
            }
        }

        const samples = this.floatTo16BitPCM(channelData);
        const dataLength = samples.length * 2;
        const bufferLength = 44 + dataLength;
        const arrayBuffer = new ArrayBuffer(bufferLength);
        const view = new DataView(arrayBuffer);

        // Write WAV header
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        this.writeString(view, 8, 'WAVE');
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, format, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numberOfChannels * (bitDepth / 8), true); // byte rate
        view.setUint16(32, numberOfChannels * (bitDepth / 8), true); // block align
        view.setUint16(34, bitDepth, true);
        this.writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        // Write PCM samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            view.setInt16(offset, samples[i], true);
        }

        return arrayBuffer;
    }

    /**
     * Convert Float32Array to 16-bit PCM
     */
    floatTo16BitPCM(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }

    /**
     * Write string to DataView
     */
    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /**
     * Convert Blob to Base64 string
     */
    async blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                // Remove data URL prefix (e.g., "data:audio/wav;base64,")
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    /**
     * Upload recording to server
     */
    async uploadToServer(textId, audioBlob) {
        try {
            // Convert to Base64
            const base64Audio = await this.blobToBase64(audioBlob);

            // Create form data
            const formData = new FormData();
            formData.append('text_id', textId);
            formData.append('base64_audio', base64Audio);

            // Upload to server
            const response = await fetch('/recordings/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '録音の保存に失敗しました');
            }

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Upload failed:', error);
            throw error;
        }
    }

    /**
     * Cancel recording without saving
     */
    cancel() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            this.isRecording = false;
            this.audioChunks = [];
        }
    }

    /**
     * Check if browser supports audio recording
     */
    static isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }
}

// Export for use in HTML
window.AudioRecorder = AudioRecorder;
