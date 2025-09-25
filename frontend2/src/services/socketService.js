import { io } from 'socket.io-client';

class SocketService {
  constructor() {
    this.socket = null;
    this.userData = null;
    this.roomData = null;
    this.isConnected = false;
  }

  connect(serverUrl = 'http://localhost:8000') {
    if (this.socket && this.socket.connected) {
      console.log('Socket already connected');
      return;
    }

    this.socket = io(serverUrl, {
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('Connected to socket server');
      this.isConnected = true;
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from socket server');
      this.isConnected = false;
    });

    this.socket.on('connected', (data) => {
      console.log('Setup completed:', data);
      this.userData = data;
    });

    this.socket.on('room joined', (data) => {
      console.log('Room joined:', data);
      this.roomData = data;
    });

    // REMOVE THESE DUPLICATE LISTENERS - they're interfering with component listeners
    // this.socket.on('processing started', (data) => {
    //   console.log('=== PROCESSING STARTED EVENT RECEIVED ===', data);
    // });

    // this.socket.on('processing progress', (data) => {
    //   console.log('=== PROCESSING PROGRESS EVENT RECEIVED ===', data);
    // });

    // this.socket.on('processing complete', (data) => {
    //   console.log('=== PROCESSING COMPLETE EVENT RECEIVED ===', data);
    // });

    // this.socket.on('processing error', (data) => {
    //   console.log('=== PROCESSING ERROR EVENT RECEIVED ===', data);
    // });

    return this.socket;
  }

  setupUser(userData = {}) {
    if (!this.socket) {
      console.error('Socket not connected');
      return;
    }
    
    this.socket.emit('setup', userData);
  }

  joinRoom(roomId = null) {
    if (!this.socket || !this.userData) {
      console.error('Socket not connected or user not setup');
      return;
    }

    const roomData = {
      userId: this.userData.userId,
      roomId: roomId
    };

    this.socket.emit('join room', roomData);
  }

  validateTrim(newTrim, leader, room) {
    if (!this.socket) {
      console.error('Socket not connected');
      return;
    }

    this.socket.emit('validate', {
      leader,
      newTrim,
      room
    });
  }

  updateTrim(newTrim, userId) {
    if (!this.socket) {
      console.error('Socket not connected');
      return;
    }

    this.socket.emit('update', {
      newTrim,
      userId
    });
  }

  onValidate(callback) {
    if (!this.socket) return;
    this.socket.on('validate', callback);
  }

  onUpdate(callback) {
    if (!this.socket) return;
    this.socket.on('update', callback);
  }

  offValidate() {
    if (!this.socket) return;
    this.socket.off('validate');
  }

  offUpdate() {
    if (!this.socket) return;
    this.socket.off('update');
  }

  processVideo(videoUrl, trimHistory, fileName = 'video') {
    if (!this.socket || !this.userData) {
      console.error('Socket not connected or user not setup');
      return;
    }

    const requestData = {
      videoUrl,
      trimHistory,
      fileName,
      userId: this.userData.userId
    };

    console.log('=== SENDING PROCESS VIDEO REQUEST ===');
    console.log('Socket ID:', this.socket.id);
    console.log('Request data:', requestData);
    console.log('Socket connected:', this.socket.connected);

    this.socket.emit('process video', requestData);
    console.log('Process video event emitted');
  }

  // Event listeners for video processing
  onProcessingStarted(callback) {
    if (!this.socket) return;
    console.log('Setting up processing started listener');
    this.socket.on('processing started', (data) => {
      console.log('Processing started callback triggered with:', data);
      callback(data);
    });
  }

  onProcessingProgress(callback) {
    if (!this.socket) return;
    console.log('Setting up processing progress listener');
    this.socket.on('processing progress', (data) => {
      console.log('Processing progress callback triggered with:', data);
      callback(data);
    });
  }

  onProcessingComplete(callback) {
    if (!this.socket) return;
    console.log('Setting up processing complete listener');
    this.socket.on('processing complete', (data) => {
      console.log('Processing complete callback triggered with:', data);
      callback(data);
    });
  }

  onProcessingError(callback) {
    if (!this.socket) return;
    console.log('Setting up processing error listener');
    this.socket.on('processing error', (data) => {
      console.log('Processing error callback triggered with:', data);
      callback(data);
    });
  }

  onProcessedVideoMetadata(callback) {
    if (!this.socket) return;
    console.log('Setting up processed video metadata listener');
    this.socket.on('processed video metadata', (data) => {
      console.log('Processed video metadata callback triggered with:', data);
      callback(data);
    });
  }

  onProcessedVideoChunk(callback) {
    if (!this.socket) return;
    console.log('Setting up processed video chunk listener');
    this.socket.on('processed video chunk', (data) => {
      callback(data);
    });
  }

  offProcessingEvents() {
    if (!this.socket) return;
    console.log('Removing processing event listeners');
    this.socket.off('processing started');
    this.socket.off('processing progress');
    this.socket.off('processing complete');
    this.socket.off('processing error');
    this.socket.off('processed video metadata');
    this.socket.off('processed video chunk');
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.userData = null;
      this.roomData = null;
      this.isConnected = false;
    }
  }

  getUserData() {
    return this.userData;
  }

  getRoomData() {
    return this.roomData;
  }

  isLeader() {
    return this.roomData && this.userData && this.roomData.leader === this.userData.userId;
  }
}

export default new SocketService();