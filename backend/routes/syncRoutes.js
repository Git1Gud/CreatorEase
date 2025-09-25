const express = require('express');
const router = express.Router();
const multer = require('multer');
const syncController = require('../controllers/syncController');

const upload = multer({ dest: 'uploads/' });

router.post('/sync-audio', upload.fields([{ name: 'video' }, { name: 'audio' }]), syncController.syncAudio);

module.exports = router;
