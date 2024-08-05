import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';

interface FaceRecognitionProps {
  onAttendanceMarked: (studentName: string) => void;
  onFaceNotDetected: () => void;
  onFaceDetected: (studentName: string) => void;
}

const FaceRecognition: React.FC<FaceRecognitionProps> = ({ onAttendanceMarked, onFaceNotDetected, onFaceDetected }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState(false);

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = '/models';
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
      setModelsLoaded(true);
    };

    loadModels();
  }, []);

  useEffect(() => {
    if (modelsLoaded && videoRef.current) {
      startVideo();
    }
  }, [modelsLoaded]);

  const startVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: {} })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error(err));
  };

  const handleVideoPlay = async () => {
    const labeledDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);

    const lastMarkedTime: { [key: string]: number } = {};

    const intervalId = setInterval(async () => {
      if (videoRef.current && canvasRef.current) {
        const detections = await faceapi
          .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, {
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight,
        });

        if (canvasRef.current) {
          canvasRef.current.innerHTML = ''; // Clear previous canvas content
          faceapi.matchDimensions(canvasRef.current, {
            width: videoRef.current.videoWidth,
            height: videoRef.current.videoHeight,
          });

          const results = resizedDetections.map((d) =>
            faceMatcher.findBestMatch(d.descriptor)
          );

          let faceDetected = false;

          results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });

            if (canvasRef.current) {
              drawBox.draw(canvasRef.current);
            }

            const studentName = result.label;
            const now = Date.now();

            if (studentName !== 'unknown') {
              faceDetected = true;
              if (!lastMarkedTime[studentName] || now - lastMarkedTime[studentName] > 60000) {
                lastMarkedTime[studentName] = now;
                onAttendanceMarked(studentName);
              }
            }
          });

          if (!faceDetected) {
            onFaceNotDetected();
          }
        }
      }
    }, 1000);

    return () => clearInterval(intervalId);
  };

  const loadLabeledImages = async () => {
    const labels = ['Student1', 'Student2'];
    return Promise.all(
      labels.map(async (label) => {
        const descriptions: Float32Array[] = [];
        for (let i = 1; i <= 2; i++) {
          try {
            const imagePath = `/labeled_images/${label}/${i}.jpg`;
            console.log(`Fetching image from: ${imagePath}`);
            const img = await faceapi.fetchImage(imagePath);
            const detections = await faceapi
              .detectSingleFace(img)
              .withFaceLandmarks()
              .withFaceDescriptor();
            if (detections?.descriptor) {
              descriptions.push(detections.descriptor);
            }
          } catch (error) {
            console.error(`Error fetching image for ${label}:`, error);
          }
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      })
    );
  };

  return (
    <div>
      <video ref={videoRef} onPlay={handleVideoPlay} autoPlay muted width="720" height="560" />
      <canvas ref={canvasRef} />
    </div>
  );
};

export default FaceRecognition;
