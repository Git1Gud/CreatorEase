import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useVideoEditor } from '../context/VideoEditorContext';
import '../styles/CollaborationSetup.css';

const CollaborationSetup = () => {
  const navigate = useNavigate();
  const {
    isConnected,
    userData,
    roomData,
    connectSocket,
    joinRoom
  } = useVideoEditor();

  const [roomInput, setRoomInput] = useState('');
  const [userName, setUserName] = useState('');

  // Redirect to edit page if already in a room
  useEffect(() => {
    if (roomData) {
      navigate('/edit');
    }
  }, [roomData, navigate]);

  const handleConnect = () => {
    if (!userName.trim()) {
      alert('Please enter your name');
      return;
    }
    connectSocket({ name: userName });
  };

  const handleJoinRoom = () => {
    const roomId = roomInput.trim() || null;
    joinRoom(roomId);
  };

  const handleCreateRoom = () => {
    joinRoom(); // Passing null creates a new room
  };

  if (!isConnected) {
    return (
      <div className="collaboration-setup-container">
        <div className="collaboration-setup-card">
          <div className="card-header">
            <h2>Join Collaboration</h2>
            <p className="subtitle">Connect with your team to edit videos together</p>
          </div>
          
          <div className="form-group">
            <label className="input-label">Your Name</label>
            <input
              type="text"
              placeholder="Enter your name"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              className="collaboration-input"
              onKeyPress={(e) => e.key === 'Enter' && handleConnect()}
            />
            <button onClick={handleConnect} className="btn-primary">
              <span className="btn-icon">→</span>
              Connect to Server
            </button>
          </div>

          <button 
            onClick={() => navigate('/')} 
            className="btn-back"
          >
            ← Back to Upload
          </button>
        </div>
      </div>
    );
  }

  if (isConnected && !roomData) {
    return (
      <div className="collaboration-setup-container">
        <div className="collaboration-setup-card">
          <div className="card-header">
            <div className="status-badge-connected">
              <span className="status-dot-pulse"></span>
              Connected
            </div>
            <h2>Welcome, {userData?.name || 'User'}!</h2>
            <p className="user-id">
              <span className="label">User ID:</span> 
              <span className="value">{userData?.userId}</span>
            </p>
          </div>
          
          <div className="room-options">
            <div className="options-divider">
              <span className="divider-line"></span>
              <span className="divider-text">Choose an option</span>
              <span className="divider-line"></span>
            </div>
            
            <div className="room-cards">
              <div className="room-card room-card-join">
                <div className="room-card-header">
                  <h3>Join Existing Room</h3>
                  <p>Collaborate on an ongoing project</p>
                </div>
                <div className="form-group">
                  <label className="input-label">Room ID</label>
                  <input
                    type="text"
                    placeholder="Enter Room ID"
                    value={roomInput}
                    onChange={(e) => setRoomInput(e.target.value)}
                    className="collaboration-input"
                    onKeyPress={(e) => e.key === 'Enter' && handleJoinRoom()}
                  />
                </div>
                <button onClick={handleJoinRoom} className="btn-card btn-card-join">
                  Join Room →
                </button>
              </div>

              <div className="room-card room-card-create">
                <div className="room-card-header">
                  <h3>Create New Room</h3>
                  <p>Start a fresh collaboration session</p>
                </div>
                <div className="room-card-content">
                  <p className="room-card-info">You'll be the room leader and can invite others to join</p>
                </div>
                <button onClick={handleCreateRoom} className="btn-card btn-card-create">
                  Create Room →
                </button>
              </div>
            </div>
          </div>

          <button 
            onClick={() => navigate('/')} 
            className="btn-back"
          >
            ← Back to Upload
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default CollaborationSetup;
