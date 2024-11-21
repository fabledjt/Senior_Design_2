// models/Answer.js
const mongoose = require('mongoose');

const answerSchema = new mongoose.Schema({
  answerNumber: Number,
  imageName: String,
  userAnswer: String,
  correctAnswer: String,
  isCorrect: Boolean,
});

const Answer = mongoose.model('Answer', answerSchema);

module.exports = Answer;
