// client/src/pages/QuizPage.js
import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "./QuizPage.css"; // Import the CSS file
import axios from 'axios';

const images = [
  { src: "/images/bread1.png", type: "bread" },
  { src: "/images/bread2.png", type: "bread" },
  // ... add all your images here
  { src: "/images/cheese12.png", type: "cheese" },
];

function QuizPage() {
  const [currentImage, setCurrentImage] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const [userAnswers, setUserAnswers] = useState(location.state?.userAnswers || []);

  useEffect(() => {
    selectRandomImage();
  }, []);

  const selectRandomImage = () => {
    const randomImage = images[Math.floor(Math.random() * images.length)];
    setCurrentImage(randomImage);
  };

  const handleAnswer = (answer) => {
    const newAnswer = {
      image: currentImage.src,
      correctAnswer: currentImage.type,
      userAnswer: answer,
    };
    axios.post('/image_transforms', { img_file: currentImage.src, user_answer: answer})
        .then(response => {
            // Handle success (optional)
            console.log(response.data);
        }).catch(error => {
            // Handle error (optional)
            console.error(error);
        }); 
    setUserAnswers((prevAnswers) => [...prevAnswers, newAnswer]);
    selectRandomImage();
  };

  const handleFinishQuiz = async () => {
    navigate("/menu", { state: { userAnswers } });
  };

  const handleRestartQuiz = () => {
    navigate("/"); // Navigate back to the start page to restart the quiz
  };

  if (!currentImage) return <div>Loading...</div>;

  return (
    <div className="quiz-container">
      <div className="content">
        <h1>Is this Cheese or Bread?</h1>
        <img src={currentImage.src} alt="Quiz" width={300} className="quiz-image" />
        <div className="button-group">
          <button className="quiz-button" onClick={() => handleAnswer("bread")}>Bread</button>
          <button className="quiz-button" onClick={() => handleAnswer("cheese")}>Cheese</button>
        </div>
        <button className="done-button" onClick={handleFinishQuiz}>Done</button>
        <h4>
          <button className="quizreturn-button" onClick={handleRestartQuiz}>
            Go to Start Quiz
          </button>
        </h4>
      </div>
    </div>
  );
}

export default QuizPage;
