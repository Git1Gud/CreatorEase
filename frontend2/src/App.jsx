import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './styles/index.css';
import './styles/App.css';
import './styles/components.css';
import './styles/Button.css';
import './styles/EditPage.css';
import './styles/UploadPage.css';
import './styles/TranscriptionPanel.css';
import UploadPage from './pages/UploadPage';
import EditPage from './pages/Editpage';
import CollaborationSetup from './pages/CollaborationSetup';
import { VideoEditorProvider } from './context/VideoEditorContext';
// import EditPage from './pages/EditPage';

const App = () => {
  return (
    <Router>
      <VideoEditorProvider>
        <div className="App">
          {/* <header className="App-header">
            <h1>Video Editor App</h1>
          </header> */}
          
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/edit" element={<EditPage />} />
              <Route path="/collaborate" element={<CollaborationSetup />} />
            </Routes>
        </div>
      </VideoEditorProvider>
    </Router>
  );
}

export default App; 