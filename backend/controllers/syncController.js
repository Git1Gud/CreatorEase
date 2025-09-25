const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const WaveFile = require('wavefile').WaveFile;

const syncAudio = async (req, res) => {
    const videoFile = req.files.video[0];
    const audioFile = req.files.audio[0];

    if (!videoFile || !audioFile) {
        return res.status(400).send('Both video and audio files are required.');
    }

    const extractedAudioFile = path.join(__dirname, '..', 'uploads', `${Date.now()}_video_audio.wav`);
    const finalOutputFile = path.join(__dirname, '..', 'outputs', `synced_${Date.now()}.mp4`);

    try {
        console.log("1. Extracting audio from video...");
        const extractResult = await extractAudioFromVideo(videoFile.path, extractedAudioFile);
        if (!extractResult.success) {
            throw new Error(`Failed to extract audio: ${extractResult.error}`);
        }

        console.log("2. Finding waveform match...");
        const offsetResult = await findWaveformMatch(extractedAudioFile, audioFile.path);
        if (!offsetResult.success) {
            throw new Error(`Failed to find waveform match: ${offsetResult.error}`);
        }

        const offsetSeconds = offsetResult.value;
        console.log(`✓ Waveform offset detected: ${offsetSeconds.toFixed(3)} seconds`);

        console.log("3. Creating final synced video...");
        const finalResult = await createFinalSyncedVideo(videoFile.path, audioFile.path, finalOutputFile, offsetSeconds);
        if (!finalResult.success) {
            throw new Error(`Failed to create final video: ${finalResult.error}`);
        }

        const finalUrl = `${req.protocol}://${req.get('host')}/outputs/${path.basename(finalOutputFile)}`;
        res.json({
            message: 'Sync successful!',
            offset: offsetSeconds,
            outputUrl: finalUrl
        });

    } catch (error) {
        console.error(`Error during sync process: ${error.message}`);
        res.status(500).send(`Error during sync process: ${error.message}`);
    } finally {
        // Cleanup temporary files
        await fs.unlink(videoFile.path).catch(err => console.error(`Failed to delete temp video file: ${err.message}`));
        await fs.unlink(audioFile.path).catch(err => console.error(`Failed to delete temp audio file: ${err.message}`));
        if (fsSync.existsSync(extractedAudioFile)) {
            await fs.unlink(extractedAudioFile).catch(err => console.error(`Failed to delete extracted audio: ${err.message}`));
        }
    }
};

// --- Helper functions ---

async function extractAudioFromVideo(inputVideo, outputAudio) {
    try {
        if (!fsSync.existsSync(inputVideo)) {
            return { success: false, error: `Video file not found: ${inputVideo}` };
        }

        const args = [
            '-i', inputVideo,
            '-vn',
            '-ac', '1',
            '-ar', '44100',
            '-acodec', 'pcm_s16le',
            '-y',
            outputAudio
        ];

        await runFFmpeg(args);

        return fsSync.existsSync(outputAudio) 
            ? { success: true } 
            : { success: false, error: "Audio extraction failed" };
            
    } catch (error) {
        return { success: false, error: `Error extracting audio: ${error.message}` };
    }
}

async function findWaveformMatch(videoAudioFile, externalAudioFile) {
    try {
        console.log("Loading audio files for waveform matching...");
        
        const videoSamples = await loadAudioSamples(videoAudioFile);
        const externalSamples = await loadAudioSamples(externalAudioFile);
        
        console.log(`Video audio: ${videoSamples.length} samples (${(videoSamples.length/44100).toFixed(1)}s)`);
        console.log(`External audio: ${externalSamples.length} samples (${(externalSamples.length/44100).toFixed(1)}s)`);
        
        let pattern, signal, videoIsPattern;
        
        if (externalSamples.length < videoSamples.length) {
            pattern = externalSamples;
            signal = videoSamples;
            videoIsPattern = false;
            console.log("Searching for external audio pattern within video audio...");
        } else {
            pattern = videoSamples;
            signal = externalSamples;
            videoIsPattern = true;
            console.log("Searching for video audio pattern within external audio...");
        }
        
        const bestOffset = await findBestMatch(pattern, signal);
        const offsetSeconds = bestOffset / 44100.0;
        
        const finalOffset = videoIsPattern ? -offsetSeconds : offsetSeconds;
        
        return { success: true, value: finalOffset };
        
    } catch (error) {
        return { success: false, error: `Error in waveform matching: ${error.message}` };
    }
}

async function loadAudioSamples(audioFile) {
    try {
        const buffer = fsSync.readFileSync(audioFile);
        const wav = new WaveFile(buffer);
        
        console.log(`Loading ${audioFile}: ${wav.fmt.sampleRate}Hz, ${wav.fmt.numChannels} channels, ${wav.bitDepth} bit`);
        
        if (wav.bitDepth !== '16') {
            wav.toBitDepth('16');
        }
        
        if (wav.fmt.numChannels > 1) {
            wav.toMono();
        }
        
        const int16Samples = wav.getSamples(false, Int16Array);
        const samples = new Float32Array(int16Samples.length);
        
        const chunkSize = 100000;
        let maxSample = 0;
        
        for (let i = 0; i < int16Samples.length; i += chunkSize) {
            const end = Math.min(i + chunkSize, int16Samples.length);
            for (let j = i; j < end; j++) {
                const val = Math.abs(int16Samples[j] / 32768.0);
                if (val > maxSample) maxSample = val;
            }
            await new Promise(resolve => setImmediate(resolve));
        }
        
        for (let i = 0; i < int16Samples.length; i += chunkSize) {
            const end = Math.min(i + chunkSize, int16Samples.length);
            for (let j = i; j < end; j++) {
                samples[j] = (int16Samples[j] / 32768.0) / (maxSample || 1);
            }
            await new Promise(resolve => setImmediate(resolve));
        }
        
        return samples;
        
    } catch (error) {
        throw new Error(`Failed to load audio samples from ${audioFile}: ${error.message}`);
    }
}

async function findBestMatch(pattern, signal) {
    const maxPatternLength = 44100 * 10; // Max 10 seconds
    const patternLength = Math.min(pattern.length, maxPatternLength);
    const searchLength = signal.length - patternLength;
    
    if (searchLength <= 0) {
        console.log("Pattern is too long for signal, using full correlation");
        return 0;
    }
    
    console.log(`Searching for best match using ${(patternLength/44100).toFixed(1)}s of audio...`);
    
    let bestOffset = 0;
    let bestCorrelation = -Infinity;
    const step = 441; // 10ms steps
    
    for (let offset = 0; offset <= searchLength; offset += step) {
        const correlation = calculateCorrelation(pattern, signal, offset, patternLength);
        
        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestOffset = offset;
        }
        
        if (offset % (44100 * 5) === 0) {
            const progress = searchLength > 0 ? (offset / searchLength * 100) : 100;
            process.stdout.write(`\rCoarse search: ${progress.toFixed(0)}% - Best at ${(bestOffset/44100).toFixed(3)}s (corr: ${bestCorrelation.toFixed(4)})`);
            await new Promise(resolve => setImmediate(resolve));
        }
    }
    
    console.log();
    
    console.log("Fine-tuning the match...");
    const fineStart = Math.max(0, bestOffset - 22050);
    const fineEnd = Math.min(searchLength, bestOffset + 22050);
    
    for (let offset = fineStart; offset <= fineEnd; offset++) {
        const correlation = calculateCorrelation(pattern, signal, offset, patternLength);
        
        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestOffset = offset;
        }
        
        if ((offset - fineStart) % 4410 === 0) {
            await new Promise(resolve => setImmediate(resolve));
        }
    }
    
    console.log(`✓ Best match found at ${(bestOffset/44100).toFixed(3)}s with correlation ${bestCorrelation.toFixed(4)}`);
    
    return bestOffset;
}

function calculateCorrelation(pattern, signal, offset, length) {
    let sum = 0;
    let patternSum = 0;
    let signalSum = 0;
    let count = 0;
    
    const maxLength = Math.min(length, pattern.length, signal.length - offset);
    
    for (let i = 0; i < maxLength; i++) {
        const p = pattern[i];
        const s = signal[i + offset];
        
        if (!isNaN(p) && !isNaN(s)) {
            sum += p * s;
            patternSum += p * p;
            signalSum += s * s;
            count++;
        }
    }
    
    if (count === 0) return 0;
    
    const denominator = Math.sqrt(patternSum) * Math.sqrt(signalSum);
    return denominator > 0 ? sum / denominator : 0;
}

async function createFinalSyncedVideo(inputVideo, inputAudio, outputVideo, offsetSeconds) {
    try {
        if (!fsSync.existsSync(inputVideo) || !fsSync.existsSync(inputAudio)) {
            return { success: false, error: "Video or audio file not found" };
        }

        let args;
        
        if (Math.abs(offsetSeconds) < 0.01) {
            args = [
                '-i', inputVideo,
                '-i', inputAudio,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y', outputVideo
            ];
        } else if (offsetSeconds > 0) {
            args = [
                '-i', inputVideo,
                '-itsoffset', offsetSeconds.toFixed(3),
                '-i', inputAudio,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y', outputVideo
            ];
        } else {
            args = [
                '-itsoffset', Math.abs(offsetSeconds).toFixed(3),
                '-i', inputVideo,
                '-i', inputAudio,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y', outputVideo
            ];
        }

        console.log(`Applying ${offsetSeconds.toFixed(3)}s offset...`);
        console.log(`Command: ffmpeg ${args.join(' ')}`);
        
        await runFFmpeg(args);

        return fsSync.existsSync(outputVideo) 
            ? { success: true } 
            : { success: false, error: "Final video creation failed" };
            
    } catch (error) {
        return { success: false, error: `Error creating final video: ${error.message}` };
    }
}

function runFFmpeg(args) {
    return new Promise((resolve, reject) => {
        const process = spawn('ffmpeg', args, {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        let stderr = '';
        
        process.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        process.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`FFmpeg failed with code ${code}: ${stderr}`));
            } else {
                resolve();
            }
        });

        process.on('error', (err) => {
            reject(err);
        });
    });
}

module.exports = {
    syncAudio
};
