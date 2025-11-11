import React from 'react';
import { useVideoEditor } from '../context/VideoEditorContext';

const CollaborationPanel = () => {
  const {
    userData,
    roomData,
    isLeader,
    disconnect
  } = useVideoEditor();

  // Only show if in a room
  if (!roomData) {
    return null;
  }

  return (
    <div className="collaboration-status">
      <div className="collaboration-info">
        <span className="status-badge">
          <span className="status-dot"></span>
          Collaboration Active
        </span>
        <span className="room-info">
          <strong>Room:</strong> {roomData?.roomId}
        </span>
        <span className="role-info">
          <strong>Role:</strong> {isLeader ? '👑 Leader' : '👤 Collaborator'}
        </span>
        <span className="user-info">
          <strong>You:</strong> {userData?.name || 'User'}
        </span>
        <span className="leader-info">
          <strong>Leader ID:</strong> {roomData?.leader}
        </span>
      </div>
      
      <button 
        onClick={disconnect}
        className="disconnect-btn"
      >
        Disconnect
      </button>
    </div>
  );
};

export default CollaborationPanel;