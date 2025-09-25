import 'dotenv/config'
import { app } from './app.js';
import { Server } from 'socket.io';
import { spawn, exec } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import https from 'https';
import http from 'http';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let currId = 1;
let currRoomId = 1;
const rooms = new Map(); // userId -> roomId
const leaders = new Map(); // roomId -> leaderId
const roomMembers = new Map(); // roomId -> Set of userIds
const roomContent = new Map(); // roomId -> { transcription, videoUrl }
const STREAM_CHUNK_SIZE = 64 * 1024; // 64KB chunks when streaming processed video over socket

const server = app.listen(process.env.PORT || 8000, () => {
    console.log(` Server is running at port : 8000 `);
});

const io = new Server(server, {
    pingTimeout: 60000,
    cors: {
        origin: "*",  
    },
});

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
const processedDir = path.join(__dirname, 'processed');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}
if (!fs.existsSync(processedDir)) {
    fs.mkdirSync(processedDir, { recursive: true });
}

app.get('/download/:fileName', (req, res) => {
    const requestedFile = req.params.fileName;
    if (!requestedFile) {
        return res.status(400).json({ error: 'Missing file name' });
    }

    const filePath = path.join(processedDir, requestedFile);
    if (!fs.existsSync(filePath)) {
        return res.status(404).json({ error: 'File not found' });
    }

    res.download(filePath, requestedFile, (err) => {
        if (err) {
            console.error('Error sending download:', err);
            if (!res.headersSent) {
                res.sendStatus(500);
            }
        }
    });
});

io.on("connection", (socket) => {
    console.log("Connected to socket.io");
    
    socket.on("setup", (userData) => {
        console.log("came to setup");
        
        if (userData?.userId) {
            console.log("existing user:", userData.userId);
            socket.join(userData.userId);
        } else {
            console.log("new user assigned ID:", currId);
            socket.join(currId);
            userData.userId = currId++;
        }

        socket.emit("connected", { userId: userData.userId, socketId: socket.id });
    });

    socket.on("join room", (room) => {
        console.log("came to join room - userId:", room.userId, "roomId:", room.roomId);
        
        if (!room.roomId) {
            // Create new room - user becomes leader
            const newRoomId = currRoomId++;
            rooms.set(room.userId, newRoomId);
            leaders.set(newRoomId, room.userId);
            
            // Initialize room members set
            if (!roomMembers.has(newRoomId)) {
                roomMembers.set(newRoomId, new Set());
            }
            roomMembers.get(newRoomId).add(room.userId);
            
            socket.join(newRoomId);
            console.log("Created new room:", newRoomId, "with leader:", room.userId);
            
            socket.emit("room joined", {
                roomId: newRoomId,
                leader: room.userId,
                isLeader: true,
                members: Array.from(roomMembers.get(newRoomId)),
                content: null
            });
            
        } else {
            // Join existing room
            const targetRoomId = parseInt(room.roomId);
            
            if (!leaders.has(targetRoomId)) {
                console.log("invalid room id:", targetRoomId);
                socket.emit("room join error", { 
                    error: "Room does not exist",
                    roomId: targetRoomId 
                });
                return;
            }
            
            // Add user to existing room
            rooms.set(room.userId, targetRoomId);
            
            // Add to room members
            if (!roomMembers.has(targetRoomId)) {
                roomMembers.set(targetRoomId, new Set());
            }
            roomMembers.get(targetRoomId).add(room.userId);
            
            socket.join(targetRoomId);
            
            const roomLeader = leaders.get(targetRoomId);
            const existingContent = roomContent.get(targetRoomId) || null;
            
            console.log("User", room.userId, "joined existing room:", targetRoomId, "with leader:", roomLeader);
            
            socket.emit("room joined", {
                roomId: targetRoomId,
                leader: roomLeader,
                isLeader: false,
                members: Array.from(roomMembers.get(targetRoomId)),
                content: existingContent
            });
            
            // Notify other room members about new user
            socket.to(targetRoomId).emit("user joined room", {
                userId: room.userId,
                members: Array.from(roomMembers.get(targetRoomId))
            });
        }
    });

    // Save room content
    socket.on("save room content", ({ roomId, transcription, videoUrl }) => {
        console.log("Saving content for room:", roomId);
        
        if (!roomId || (!transcription && !videoUrl)) {
            console.log("Invalid room content data");
            socket.emit("content save error", { error: "Invalid data" });
            return;
        }

        const userId = Array.from(rooms.entries()).find(([uid, rid]) => rid === roomId)?.[0];
        const roomLeader = leaders.get(roomId);
        
        if (!userId || userId !== roomLeader) {
            console.log("Only room leader can save content");
            socket.emit("content save error", { error: "Only room leader can save content" });
            return;
        }

        const existingContent = roomContent.get(roomId) || {};
        const updatedContent = {
            ...existingContent,
            ...(transcription && { transcription }),
            ...(videoUrl && { videoUrl }),
            lastUpdated: new Date().toISOString()
        };

        roomContent.set(roomId, updatedContent);
        
        console.log("Content saved for room", roomId);
        
        socket.to(roomId).emit("room content updated", {
            roomId,
            content: updatedContent
        });
        
        socket.emit("content saved", {
            roomId,
            content: updatedContent
        });
    });

    // Get room content
    socket.on("get room content", ({ roomId }) => {
        console.log("Getting content for room:", roomId);
        
        const content = roomContent.get(roomId) || null;
        
        socket.emit("room content", {
            roomId,
            content
        });
    });

    // NEW: Process video with trim history using system FFmpeg
   // In your socket event handler, fix the input path
socket.on("process video", async ({ videoUrl, trimHistory, userId, fileName }) => {
    console.log("Processing video for user:", userId);
    console.log("Trim history:", trimHistory);
    console.log("Video URL:", videoUrl);

    try {
        socket.emit("processing started", { 
            message: "Starting video processing...",
            progress: 0 
        });

        let inputVideoPath=path.join(__dirname, '2x.mp4');
        
        // if (videoUrl.startsWith('http')) {
        //     inputVideoPath = await downloadVideoFile(videoUrl, userId);
        // } else if (videoUrl.startsWith('blob:')) {
        //     // Handle blob URLs - for now, use a default video file
        //     inputVideoPath = path.join(__dirname, '2x.mp4'); // Make sure this path is correct
        //     console.log("Using local video file:", inputVideoPath);
        // } else {
        //     // Assume it's a local path, but make it absolute
        //     inputVideoPath = path.resolve(videoUrl);
        // }

        // Check if input file exists
        if (!fs.existsSync(inputVideoPath)) {
            throw new Error(`Input video file not found: ${inputVideoPath}`);
        }

        socket.emit("processing progress", { 
            message: "Video found, processing trims...",
            progress: 10 
        });

        // Process video with trim history using system FFmpeg
        const outputPath = await processVideoWithFFmpeg(inputVideoPath, trimHistory, userId, fileName, (progress) => {
            socket.emit("processing progress", { 
                message: `Processing video... ${Math.round(progress)}% complete`,
                progress: 10 + (progress * 0.9) // 10% already done, remaining 90% for processing
            });
        });

        socket.emit("processing progress", {
            message: "Processing complete, preparing video stream...",
            progress: 95
        });

        const streamedMeta = await streamProcessedVideoOverSocket(socket, outputPath, (fraction) => {
            socket.emit("processing progress", {
                message: "Streaming processed video...",
                progress: Math.min(95 + Math.round(fraction * 5), 100)
            });
        });
        
        socket.emit("processing complete", { 
            success: true,
            fileName: streamedMeta.fileName,
            size: streamedMeta.size,
            mimeType: streamedMeta.mimeType,
            downloadUrl: `/download/${path.basename(outputPath)}`,
            message: "Video processed successfully and delivered via socket."
        });

    } catch (error) {
        console.error("Error processing video:", error);
        socket.emit("processing error", { 
            error: "Failed to process video",
            details: error.message 
        });
    }
});

    // Existing socket events (validate, update, etc.)
    socket.on("validate", ({ leader, newTrim, room }) => {
        console.log("Validation request - leader:", leader, "room:", room);
        
        if (!leader || !newTrim || !room) {
            console.log("invalid validation data");
            return;
        }
        
        socket.to(leader).emit("validate", { newTrim });
    });

    socket.on("update", ({ newTrim, userId }) => {
        console.log("Broadcasting update from user:", userId);
        
        const userRoomId = rooms.get(userId);
        if (userRoomId) {
            socket.to(userRoomId).emit("update", { newTrim });
            console.log("Update broadcasted to room:", userRoomId);
        } else {
            console.log("User not in any room:", userId);
        }
    });

    socket.on("disconnect", () => {
        console.log("USER DISCONNECTED");
    });
});

// Helper function to download video from URL using Node.js built-in modules
function downloadVideoFile(videoUrl, userId) {
    return new Promise((resolve, reject) => {
        const fileName = `input_${userId}_${Date.now()}.mp4`;
        const outputPath = path.join(uploadsDir, fileName);
        const file = fs.createWriteStream(outputPath);
        
        console.log(`Downloading video from ${videoUrl} to ${outputPath}`);
        
        const client = videoUrl.startsWith('https:') ? https : http;
        
        const request = client.get(videoUrl, (response) => {
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download video: ${response.statusCode}`));
                return;
            }
            
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                console.log('Video download complete');
                resolve(outputPath);
            });
        });
        
        request.on('error', (err) => {
            fs.unlink(outputPath, () => {}); // Delete the file on error
            reject(err);
        });
        
        file.on('error', (err) => {
            fs.unlink(outputPath, () => {}); // Delete the file on error
            reject(err);
        });
    });
}

// Helper function to process video with trim history using system FFmpeg
// Helper function to process video with trim history using system FFmpeg
// Updated processVideoWithFFmpeg function using single filter_complex approach
function processVideoWithFFmpeg(inputPath, trimHistory, userId, fileName, progressCallback) {
    return new Promise((resolve, reject) => {
        const sanitizedFileName = (fileName || 'video').replace(/[^a-zA-Z0-9_-]/g, '_');
        const outputFileName = `processed_${sanitizedFileName}_${userId}_${Date.now()}.mp4`;
        const outputPath = path.join(processedDir, outputFileName);

        console.log("Input path:", inputPath);
        console.log("Output path:", outputPath);
        console.log("Trim history:", trimHistory);

        if (!trimHistory || trimHistory.length === 0) {
            console.log("No trims to apply, re-encoding for compatibility");
            
            const copyArgs = [
                '-i', inputPath,
                '-c:v', 'libx264',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-ac', '2',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-shortest',
                '-y',
                outputPath
            ];
            
            console.log('FFmpeg no-trim command:', 'ffmpeg', copyArgs.join(' '));
            const ffmpegProcess = spawn('ffmpeg', copyArgs);
            
            let stderr = '';
            ffmpegProcess.stderr.on('data', (data) => {
                stderr += data.toString();
                
                // Extract progress
                const timeMatch = data.toString().match(/time=(\d{2}):(\d{2}):(\d{2})\./);
                if (timeMatch && progressCallback) {
                    progressCallback(50); // Simple progress for copy
                }
            });
            
            ffmpegProcess.on('close', (code) => {
                if (code === 0 && fs.existsSync(outputPath)) {
                    const stats = fs.statSync(outputPath);
                    if (stats.size > 0) {
                        console.log(`Output file created: ${outputPath} (${stats.size} bytes)`);
                        resolve(outputPath);
                    } else {
                        reject(new Error('Output file is empty'));
                    }
                } else {
                    console.error('FFmpeg stderr:', stderr);
                    reject(new Error(`FFmpeg process failed with code ${code}`));
                }
            });
            
            ffmpegProcess.on('error', reject);
            return;
        }

        // Process with trims using single filter_complex approach
        const sortedTrims = trimHistory.sort((a, b) => a.start - b.start);
        console.log("Processing trims:", sortedTrims);

        // Create segments to keep (similar to your diarization logic)
        const segments = createSegmentsToKeep(sortedTrims);
        console.log("Segments to keep:", segments);

        if (segments.length === 0) {
            reject(new Error("All video content was trimmed"));
            return;
        }

        // Create single filter_complex command
        const filterComplex = createFilterComplexForSegments(segments);
        console.log("Filter complex:", filterComplex);

        // Build single FFmpeg command
        const args = [
            '-i', inputPath,
            '-filter_complex', filterComplex,
            '-map', '[final_video]',
            '-map', '[final_audio]',
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-fflags', '+genpts',
            '-y',
            outputPath
        ];

        console.log('FFmpeg filter_complex command:', 'ffmpeg', args.join(' '));

        const ffmpegProcess = spawn('ffmpeg', args);
        let stderr = '';
        let duration = 0;

        ffmpegProcess.stderr.on('data', (data) => {
            stderr += data.toString();
            // console.log('FFmpeg stderr:', data.toString());
            
            // Extract duration for progress calculation
            if (duration === 0) {
                const durationMatch = data.toString().match(/Duration: (\d{2}):(\d{2}):(\d{2})\./);
                if (durationMatch) {
                    const hours = parseInt(durationMatch[1]);
                    const minutes = parseInt(durationMatch[2]);
                    const seconds = parseInt(durationMatch[3]);
                    duration = hours * 3600 + minutes * 60 + seconds;
                }
            }
            
            // Extract progress
            const timeMatch = data.toString().match(/time=(\d{2}):(\d{2}):(\d{2})\./);
            if (timeMatch && duration > 0 && progressCallback) {
                const hours = parseInt(timeMatch[1]);
                const minutes = parseInt(timeMatch[2]);
                const seconds = parseInt(timeMatch[3]);
                const currentTime = hours * 3600 + minutes * 60 + seconds;
                const progress = Math.min((currentTime / duration) * 100, 100);
                progressCallback(progress);
            }
        });

        ffmpegProcess.on('close', (code) => {
            console.log(`FFmpeg process exited with code: ${code}`);
            
            if (code === 0) {
                if (fs.existsSync(outputPath)) {
                    const stats = fs.statSync(outputPath);
                    if (stats.size > 0) {
                        console.log(`Final output created: ${outputPath} (${stats.size} bytes)`);
                        
                        // Clean up input file if downloaded
                        if (inputPath.includes('uploads/input_')) {
                            fs.unlink(inputPath, (err) => {
                                if (err) console.error('Error cleaning up input file:', err);
                            });
                        }
                        
                        resolve(outputPath);
                    } else {
                        reject(new Error('Final output file is empty'));
                    }
                } else {
                    reject(new Error('Final output file was not created'));
                }
            } else {
                console.error('FFmpeg stderr:', stderr);
                reject(new Error(`FFmpeg process failed with code ${code}`));
            }
        });

        ffmpegProcess.on('error', reject);
    });
}

function streamProcessedVideoOverSocket(socket, outputPath, progressCallback) {
    return new Promise((resolve, reject) => {
        fs.stat(outputPath, (statErr, stats) => {
            if (statErr) {
                reject(statErr);
                return;
            }

            const fileName = path.basename(outputPath);
            const totalSize = stats.size;
            const mimeType = 'video/mp4';

            socket.emit("processed video metadata", {
                fileName,
                size: totalSize,
                mimeType
            });

            if (totalSize === 0) {
                if (progressCallback) {
                    progressCallback(1);
                }

                socket.emit("processed video chunk", {
                    fileName,
                    index: 0,
                    chunk: null,
                    isLast: true
                });

                resolve({ fileName, size: totalSize, mimeType });
                return;
            }

            const readStream = fs.createReadStream(outputPath, { highWaterMark: STREAM_CHUNK_SIZE });
            let bytesSent = 0;
            let chunkIndex = 0;

            readStream.on('data', (chunk) => {
                bytesSent += chunk.length;

                if (progressCallback) {
                    progressCallback(Math.min(bytesSent / totalSize, 1));
                }

                socket.emit("processed video chunk", {
                    fileName,
                    index: chunkIndex++,
                    chunk: chunk.toString('base64'),
                    isLast: false
                });
            });

            readStream.on('end', () => {
                socket.emit("processed video chunk", {
                    fileName,
                    index: chunkIndex,
                    chunk: null,
                    isLast: true
                });

                resolve({ fileName, size: totalSize, mimeType });
            });

            readStream.on('error', (streamErr) => {
                readStream.destroy();
                reject(streamErr);
            });
        });
    });
}

// Helper function to create segments to keep (similar to your diarization logic)
function createSegmentsToKeep(sortedTrims) {
    const segments = [];
    let currentStart = 0;

    // Create segments between trims (parts to keep)
    sortedTrims.forEach((trim) => {
        if (currentStart < trim.start) {
            segments.push({
                start: currentStart,
                end: trim.start,
                duration: trim.start - currentStart
            });
        }
        currentStart = trim.end;
    });

    // Add final segment after last trim (till end of video)
    segments.push({
        start: currentStart,
        end: null, // null means till end
        duration: null // will be calculated by FFmpeg
    });

    // Filter out segments that are too short (less than 0.1 seconds)
    return segments.filter(segment => 
        segment.end === null || (segment.end - segment.start) >= 0.1
    );
}

// Helper function to create filter_complex string for segments
function createFilterComplexForSegments(segments) {
    let filterParts = [];
    let videoOutputs = [];
    let audioOutputs = [];

    // Create trim filters for each segment
    segments.forEach((segment, index) => {
        const segmentLabel = `seg${index}`;
        
        if (segment.end === null) {
            // Segment till end of video
            filterParts.push(
                `[0:v]trim=start=${segment.start.toFixed(3)},setpts=PTS-STARTPTS[v${index}]`,
                `[0:a]atrim=start=${segment.start.toFixed(3)},asetpts=PTS-STARTPTS[a${index}]`
            );
        } else {
            // Segment with specific end time
            filterParts.push(
                `[0:v]trim=start=${segment.start.toFixed(3)}:end=${segment.end.toFixed(3)},setpts=PTS-STARTPTS[v${index}]`,
                `[0:a]atrim=start=${segment.start.toFixed(3)}:end=${segment.end.toFixed(3)},asetpts=PTS-STARTPTS[a${index}]`
            );
        }
        
        videoOutputs.push(`[v${index}]`);
        audioOutputs.push(`[a${index}]`);
    });

    // Concatenate all segments
    const videoConcat = `${videoOutputs.join('')}concat=n=${segments.length}:v=1:a=0[final_video]`;
    const audioConcat = `${audioOutputs.join('')}concat=n=${segments.length}:v=0:a=1[final_audio]`;
    
    filterParts.push(videoConcat, audioConcat);

    return filterParts.join(';');
}
// Helper function to concatenate segments
function concatenateSegments(tempFiles, outputPath, progressCallback) {
    return new Promise((resolve, reject) => {
        if (tempFiles.length === 1) {
            // If only one segment, just rename it
            fs.rename(tempFiles[0], outputPath, (err) => {
                if (err) reject(err);
                else {
                    if (progressCallback) progressCallback(100);
                    resolve();
                }
            });
            return;
        }

        // Create concat file list
        const concatFilePath = path.join(path.dirname(outputPath), `concat_${Date.now()}.txt`);
        const concatContent = tempFiles.map(file => `file '${file}'`).join('\n');
        
        fs.writeFileSync(concatFilePath, concatContent);

        const args = [
            '-f', 'concat',
            '-safe', '0',
            '-i', concatFilePath,
            '-c', 'copy',
            '-y',
            outputPath
        ];

        console.log('Concatenating segments:', args.join(' '));

        const ffmpegProcess = spawn('ffmpeg', args);

        // ffmpegProcess.stderr.on('data', (data) => {
        //     console.log('Concat stderr:', data.toString());
        // });

        ffmpegProcess.on('close', (code) => {
            // Clean up concat file
            fs.unlink(concatFilePath, (err) => {
                if (err) console.error('Error cleaning up concat file:', err);
            });

            if (code === 0) {
                if (progressCallback) progressCallback(100);
                resolve();
            } else {
                reject(new Error(`Concatenation failed with code ${code}`));
            }
        });

        ffmpegProcess.on('error', reject);
    });
}