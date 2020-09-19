import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';



function App() {
  const url = {
    model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};

const PAD_INDEX = 0;
const OOV_INDEX = 2;

const [metadata, setMetadata] = useState(0);
const [model, setModel] = useState(0);

async function loadModel(url) {
  console.log(url)
    try {
        const model = await tf.loadLayersModel(url.model);
        setModel(model)
    } catch (err) {
        console.log(err);
    }
}
 
async function loadMetadata(url) {
    try {
        const metadataJson = await fetch(url.metadata);
        const metadata = await metadataJson.json();
setMetadata(metadata)    
} catch (err) {
        console.log(err);
    }
}

const padSequences = (sequences, maxLen, padding = 'pre', truncating = 'pre', value = PAD_INDEX) => {
  return sequences.map(seq => {
    if (seq.length > maxLen) {
      if (truncating === 'pre') {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; ++i) {
        pad.push(value);
      }
      if (padding === 'pre') {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }

    return seq;
  });
}

const getSentimentScore =(text) => {
  const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  const sequence = inputText.map(word => {
    let wordIndex = metadata.word_index[word] + metadata.index_from;
    if (wordIndex > metadata.vocabulary_size) {
      wordIndex = OOV_INDEX;
    }
    return wordIndex;
  });
  // Perform truncation and padding.
  const paddedSequence = padSequences([sequence], metadata.max_len);
  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  console.log(score)
  return score;
}


useEffect(()=>{
  tf.ready().then(
    ()=>{
      console.log("tf ready")
      loadModel(url)
      loadMetadata(url)
    }
  );

},[])

  return (
    <div className="App">
      <header className="App-header">
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>

      </header>
      <button onClick={()=>getSentimentScore("bad")}></button>
    </div>
  );
}

export default App;
