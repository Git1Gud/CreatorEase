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
        
        <div style={{ marginTop: '12px' }}>
          <h4>Join or Create Room</h4>
          <div>
            <input
              type="text"
              placeholder="Enter Room ID (optional)"
              value={roomInput}
              onChange={(e) => setRoomInput(e.target.value)}
            />
            <button onClick={handleJoinRoom}>
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
          background: '#2a1a1a', 
          color: '#ff6b6b',
          borderColor: '#4a2a2a'
        }}
      >
        Disconnect
      </button>
    </div>
  );
};

export default CollaborationPanel;