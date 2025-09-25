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
        
        socket.emit("processing complete", { 
            success: true,
            downloadUrl: `/download/${path.basename(outputPath)}`,
            message: "Video processed successfully! Click to download."
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
function processVideoWithFFmpeg(inputPath, trimHistory, userId, fileName, progressCallback) {
    return new Promise((resolve, reject) => {
        if (!trimHistory || trimHistory.length === 0) {
            console.log("No trims to apply, copying original file");
            const outputFileName = `processed_${fileName || 'video'}_${userId}_${Date.now()}.mp4`;
            const outputPath = path.join(processedDir, outputFileName);
            
            // Just copy the file if no trims
            const copyArgs = ['-i', inputPath, '-c', 'copy', outputPath];
            const ffmpegProcess = spawn('ffmpeg', copyArgs);
            
            ffmpegProcess.stderr.on('data', (data) => {
                console.log('FFmpeg stderr:', data.toString());
            });
            
            ffmpegProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(outputPath);
                } else {
                    reject(new Error(`FFmpeg process exited with code ${code}`));
                }
            });
            
            ffmpegProcess.on('error', reject);
            return;
        }

        // Sort trim history by start time
        const sortedTrims = trimHistory.sort((a, b) => a.start - b.start);
        
        console.log("Applying trims:", sortedTrims);

        const outputFileName = `processed_${fileName || 'video'}_${userId}_${Date.now()}.mp4`;
        const outputPath = path.join(processedDir, outputFileName);

        // Create segments between trims (keep everything except trimmed parts)
        let segments = [];
        let currentStart = 0;

        sortedTrims.forEach((trim) => {
            // Add segment before this trim (if there's content)
            if (currentStart < trim.start) {
                segments.push({
                    start: currentStart,
                    end: trim.start
                });
            }
            currentStart = trim.end;
        });

        // Add final segment after last trim - but we need to get actual video duration
        // For now, let's use a more reasonable end time
        segments.push({
            start: currentStart,
            end: null // null means till the end
        });

        // Filter out empty segments
        segments = segments.filter(segment => segment.end === null || segment.end > segment.start);

        if (segments.length === 0) {
            reject(new Error("All video content was trimmed"));
            return;
        }

        console.log("Segments to keep:", segments);

        // Create temporary files for each segment
        const tempFiles = [];
        const tempDir = path.join(processedDir, 'temp');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        let processedSegments = 0;

        // Process each segment separately, then concatenate
        const processSegment = (segmentIndex) => {
            return new Promise((segmentResolve, segmentReject) => {
                const segment = segments[segmentIndex];
                const tempFileName = `temp_${userId}_${Date.now()}_${segmentIndex}.mp4`;
                const tempFilePath = path.join(tempDir, tempFileName);
                tempFiles.push(tempFilePath);

                let args;
                if (segment.end === null) {
                    // From start to end of video
                    args = [
                        '-i', inputPath,
                        '-ss', segment.start.toString(),
                        '-c', 'copy',
                        '-avoid_negative_ts', 'make_zero',
                        '-y',
                        tempFilePath
                    ];
                } else {
                    // From start to specific end time
                    const duration = segment.end - segment.start;
                    args = [
                        '-i', inputPath,
                        '-ss', segment.start.toString(),
                        '-t', duration.toString(),
                        '-c', 'copy',
                        '-avoid_negative_ts', 'make_zero',
                        '-y',
                        tempFilePath
                    ];
                }

                console.log(`Processing segment ${segmentIndex + 1}/${segments.length}:`, args.join(' '));

                const ffmpegProcess = spawn('ffmpeg', args);

                ffmpegProcess.stderr.on('data', (data) => {
                    console.log(`Segment ${segmentIndex} stderr:`, data.toString());
                });

                ffmpegProcess.on('close', (code) => {
                    if (code === 0) {
                        processedSegments++;
                        const progress = (processedSegments / (segments.length + 1)) * 80; // 80% for segments, 20% for concat
                        if (progressCallback) progressCallback(progress);
                        segmentResolve();
                    } else {
                        segmentReject(new Error(`Segment processing failed with code ${code}`));
                    }
                });

                ffmpegProcess.on('error', segmentReject);
            });
        };

        // Process all segments sequentially
        const processAllSegments = async () => {
            try {
                for (let i = 0; i < segments.length; i++) {
                    await processSegment(i);
                }

                // Now concatenate all segments
                await concatenateSegments(tempFiles, outputPath, progressCallback);
                
                // Clean up temp files
                tempFiles.forEach(file => {
                    fs.unlink(file, (err) => {
                        if (err) console.error('Error cleaning up temp file:', err);
                    });
                });

                // Clean up input file if it was downloaded
                if (inputPath.includes('uploads/input_')) {
                    fs.unlink(inputPath, (err) => {
                        if (err) console.error('Error cleaning up input file:', err);
                    });
                }

                resolve(outputPath);
            } catch (error) {
                // Clean up temp files on error
                tempFiles.forEach(file => {
                    fs.unlink(file, () => {});
                });
                reject(error);
            }
        };

        processAllSegments();
    });
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

        ffmpegProcess.stderr.on('data', (data) => {
            console.log('Concat stderr:', data.toString());
        });

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