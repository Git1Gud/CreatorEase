const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');
const fsSync = require('fs');

const syncRoutes = require('./routes/syncRoutes');

const app = express();
const port = process.env.PORT || 4000;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));

// Ensure required directories exist
const outputDir = path.join(__dirname, 'outputs');
const uploadsDir = path.join(__dirname, 'uploads');
if (!fsSync.existsSync(outputDir)) {
    fsSync.mkdirSync(outputDir);
}
if (!fsSync.existsSync(uploadsDir)) {
    fsSync.mkdirSync(uploadsDir);
}

app.use('/api', syncRoutes);

app.get('/', (req, res) => {
    res.send('Audio Sync Server is running!');
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});