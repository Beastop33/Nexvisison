const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const compression = require('compression');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Enable compression
app.use(compression());

// Serve static files from 'public' directory
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Proxy configuration for Object Detection backend
app.use('/detect', createProxyMiddleware({
    target: 'http://localhost:5000',
    changeOrigin: true,
    pathRewrite: {
        '^/detect': '/detect'
    },
    onProxyRes: function (proxyRes, req, res) {
        proxyRes.headers['Access-Control-Allow-Origin'] = '*';
        proxyRes.headers['Cache-Control'] = 'no-cache';
    },
    proxyTimeout: 5000,
    timeout: 5000
}));

// Proxy configuration for OCR backend
app.use('/ocr', createProxyMiddleware({
    target: 'http://localhost:5001',
    changeOrigin: true,
    pathRewrite: {
        '^/ocr': '/ocr'
    },
    onProxyRes: function (proxyRes, req, res) {
        proxyRes.headers['Access-Control-Allow-Origin'] = '*';
        proxyRes.headers['Cache-Control'] = 'no-cache';
    },
    proxyTimeout: 5000,
    timeout: 5000
}));

// Routes
app.get('/sender', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'sender.html'));
});

app.get('/receiver', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'receiver.html'));
});

// WebRTC signaling
const clients = { sender: null, receiver: null };

io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);

    socket.on('register_role', (role) => {
        clients[role] = socket.id;
        socket.role = role;
        console.log(`Client ${socket.id} registered as ${role}`);
    });

    socket.on('signal', (data) => {
        const targetRole = socket.role === 'sender' ? 'receiver' : 'sender';
        if (clients[targetRole]) {
            console.log(`Forwarding signal from ${socket.role} to ${targetRole}`);
            io.to(clients[targetRole]).emit('signal', data);
        } else {
            console.log(`No client found for role: ${targetRole}`);
        }
    });

    socket.on('disconnect', () => {
        if (socket.role) {
            clients[socket.role] = null;
            console.log(`Client ${socket.id} disconnected`);
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Access sender at http://localhost:${PORT}/sender`);
    console.log(`Access receiver at http://localhost:${PORT}/receiver`);
});