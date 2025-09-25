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
// import EditPage from './pages/EditPage';

const App = () => {
  return (
    <Router>
      <div className="App">
        {/* <header className="App-header">
          <h1>Video Editor App</h1>
        </header> */}
        <main>
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/edit" element={<EditPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 