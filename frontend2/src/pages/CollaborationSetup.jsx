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
          <h2>Join Collaboration</h2>
          <p className="subtitle">Enter your name to get started</p>
          
          <div className="form-group">
            <input
              type="text"
              placeholder="Enter your name"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
              className="collaboration-input"
              onKeyPress={(e) => e.key === 'Enter' && handleConnect()}
            />
            <button onClick={handleConnect} className="btn-primary">
              Connect
            </button>
          </div>

          <button 
            onClick={() => navigate('/')} 
            className="btn-secondary"
            style={{ marginTop: '20px' }}
          >
            Back to Upload
          </button>
        </div>
      </div>
    );
  }

  if (isConnected && !roomData) {
    return (
      <div className="collaboration-setup-container">
        <div className="collaboration-setup-card">
          <h2>Welcome, {userData?.name || 'User'}</h2>
          <p className="user-id">User ID: {userData?.userId}</p>
          
          <div className="room-options">
            <h3>Join or Create Room</h3>
            
            <div className="form-group">
              <input
                type="text"
                placeholder="Enter Room ID (optional)"
                value={roomInput}
                onChange={(e) => setRoomInput(e.target.value)}
                className="collaboration-input"
                onKeyPress={(e) => e.key === 'Enter' && handleJoinRoom()}
              />
            </div>

            <div className="button-group">
              <button onClick={handleJoinRoom} className="btn-primary">
                Join Room
              </button>
              <button onClick={handleCreateRoom} className="btn-secondary">
                Create New Room
              </button>
            </div>
          </div>

          <button 
            onClick={() => navigate('/')} 
            className="btn-back"
            style={{ marginTop: '20px' }}
          >
            Back to Upload
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default CollaborationSetup;
