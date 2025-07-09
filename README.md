# Cleft Generation Demo

A web application for facial inpainting using deep learning models, built with Next.js frontend and Flask backend. (TODO: add details later)

## Project Structure

```
cleft_gen_demo/
├── app/                    # Next.js frontend
├── components/             # React components
├── scripts/               # Flask backend
│   ├── flask_backend.py   # Main Flask server
│   ├── wrapper_modified.py # Image processing wrapper
│   └── requirements.txt   # Python dependencies
├── src/                   # Model source code
├── checkpoints/           # Pre-trained model weights
├── main.py               # Model entry point
├── test.py               # Model testing script
└── requirements.txt      # Complete Python dependencies
```

## Prerequisites

- Python 3.9+
- Node.js 18+
- npm or pnpm

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Joy12343/cleft-demo-app.git
cd cleft_gen_demo
```

### 2. Set up Python environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set up Node.js environment
```bash
# Install Node.js dependencies
npm install
# or
pnpm install
```

## Running the Application

### Option 1: Run both frontend and backend separately

1. **Start the Flask backend:**
```bash
cd scripts
python flask_backend.py
```
The backend will run on `http://localhost:5000`

2. **Start the Next.js frontend:**
```bash
npm run dev
# or
pnpm dev
```
The frontend will run on `http://localhost:3000`

### Option 2: Use the startup script (if available)
```bash
./start_app.sh
```

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Upload a source image and a mask image
3. The application will process the images and display the results
4. You can download the processed images

## Dependencies

### Python Dependencies
- **Web Framework**: Flask, Flask-CORS, Werkzeug
- **Image Processing**: Pillow, OpenCV, scikit-image, imageio
- **Computer Vision**: face-alignment
- **Scientific Computing**: numpy, scipy
- **Deep Learning**: PyTorch, torchvision
- **Configuration**: PyYAML
- **Visualization**: matplotlib
- **Hyperparameter Optimization**: optuna

### Node.js Dependencies
- **Framework**: Next.js 15, React 19
- **UI Components**: Radix UI components
- **Styling**: Tailwind CSS
- **Forms**: React Hook Form, Zod
- **Utilities**: Various utility libraries

## Model Information

The application uses a deep learning model for facial inpainting:
- **Model Type**: Inpainting Generator with Discriminator
- **Architecture**: Based on NCLG (Neural Cleft Lip Generation)
- **Input**: Source image + binary mask + facial landmarks
- **Output**: Inpainted image

## API Endpoints

- `POST /api/process` - Process uploaded images
- `GET /api/download/<session_id>/<filename>` - Download processed results
- `GET /api/health` - Health check

## Development

### Backend Development
- The Flask backend is in `scripts/flask_backend.py`
- Image processing logic is in `scripts/wrapper_modified.py`
- Model code is in the `src/` directory

### Frontend Development
- Next.js app is in the `app/` directory
- Components are in the `components/` directory
- Styling uses Tailwind CSS

## Troubleshooting

1. **CUDA Issues**: The model will automatically fall back to CPU if CUDA is not available
2. **Port Conflicts**: Make sure ports 3000 and 5000 are available
3. **Memory Issues**: Large images may require more memory
4. **Face Detection**: Ensure uploaded images contain clear faces
