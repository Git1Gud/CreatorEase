import React, { useState } from 'react';
import { useVideoEditor } from '../context/VideoEditorContext';

const CollaborationPanel = () => {
  const {
    isConnected,
    userData,
    roomData,
    isLeader,
    connectSocket,
    joinRoom,
    disconnect
  } = useVideoEditor();

  const [roomInput, setRoomInput] = useState('');
  const [userName, setUserName] = useState('');

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
      <div className="collaboration-panel">
        <h3>Join Collaboration</h3>
        <div>
          <input
            type="text"
            placeholder="Enter your name"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            style={{ marginRight: '10px', padding: '5px' }}
          />
          <button onClick={handleConnect}>Connect</button>
        </div>
      </div>
    );
  }

  if (isConnected && !roomData) {
    return (
      <div className="collaboration-panel">
        <h3>Welcome, {userData?.name || 'User'}</h3>
        <p>User ID: {userData?.userId}</p>
        
        <div style={{ marginTop: '20px' }}>
          <h4>Join or Create Room</h4>
          <div>
            <input
              type="text"
              placeholder="Enter Room ID (optional)"
              value={roomInput}
              onChange={(e) => setRoomInput(e.target.value)}
              style={{ marginRight: '10px', padding: '5px' }}
            />
            <button onClick={handleJoinRoom} style={{ marginRight: '10px' }}>
              Join Room
            </button>
            <button onClick={handleCreateRoom}>
              Create New Room
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="collaboration-panel">
      <h3>Collaboration Active</h3>
      <div>
        <p><strong>Room ID:</strong> {roomData?.roomId}</p>
        <p><strong>Your Role:</strong> {isLeader ? 'Leader' : 'Collaborator'}</p>
        <p><strong>Leader ID:</strong> {roomData?.leader}</p>
      </div>
      
      <button 
        onClick={disconnect}
        style={{ 
          marginTop: '10px', 
          background: 'linear-gradient(145deg, rgba(248,113,113,0.9), rgba(244,63,94,0.78))', 
          color: 'var(--text-primary)',
          padding: '8px 16px',
          border: '1px solid rgba(248, 113, 113, 0.38)',
          borderRadius: '12px',
          cursor: 'pointer',
          boxShadow: '0 14px 28px rgba(248, 113, 113, 0.28)',
          fontWeight: 600,
          letterSpacing: '0.02em'
        }}
      >
        Disconnect
      </button>
    </div>
  );
};

export default CollaborationPanel;