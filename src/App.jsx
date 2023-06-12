import { useState, useEffect } from 'react';
import './App.css';
import { Navbar, NavbarBrand } from '@nextui-org/react';
import { Input, Button } from '@nextui-org/react';
import { Link } from 'react-router-dom';
import axios from 'axios';

function App() {
  const [query, searchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(true);
  const [showResults, setShowResults] = useState(false);
  const [emotions, setEmotions] = useState('');
  const [fade, setFade] = useState(false);
  const [fadeEmotions, setFadeEmotions] = useState(false);
  const [animateGradient, setAnimateGradient] = useState(false);
  const [backgroundStyle, setBackgroundStyle] = useState('#242424');



  const handleSearch = async (e) => {
    e.preventDefault();
    console.log("Got here and the query is " + query);

    try {
      const response = await axios.get('http://127.0.0.1:5000/predict-emotion', {
        params: {
          track_name: query
        }
      });

      console.log(response.data.emotion);
      const emotionCombination = response.data.emotion.join('-');
      const gradient = getGradientForCombination(emotionCombination);
      
      setEmotions(response.data.emotion.join(' '));
      setShowResults(true);
      setShowSearch(false);
      setAnimateGradient(true);
      setBackgroundStyle(gradient);
    
    } catch (error) {
      console.log(error);
    }
  };

  const handleChange = (e) => {
    searchQuery(e.target.value);
  };

  useEffect(() => {
    setFade(true);
    const timeout = setTimeout(() => {
      setFadeEmotions(true);
    }, 500);

    return () => clearTimeout(timeout);
  }, []);




  const handleRefresh = () => {
    window.location.reload();
  };

  const handleFadeOut = () => {
    setFade(false);
  };

  const getGradientForCombination = (combination) => {
    const gradients = {
      'happiness-sadness': 'linear-gradient(-45deg, #FFFA00, #0065FF)', 
      'happiness-calm': 'linear-gradient(-45deg, #FFFA00, #00FF71)',
      'happiness-love': 'linear-gradient(-45deg, #FFFA00, #F100FF)', 
      'happiness-energetic': 'linear-gradient(-45deg, #FFFA00, #FF8200)', 
      'sadness-calm': 'linear-gradient(-45deg, #0065FF, #00FF71)',
      'sadness-love': 'linear-gradient(-45deg, #0065FF, #F100FF)',
      'sadness-happiness': 'linear-gradient(-45deg, #0065FF, #FFFA00)',
      'sadness-energetic': 'linear-gradient(-45deg, #0065FF, #FF8200)',
      'calm-happiness': 'linear-gradient(-45deg, #00FF71, #FFFA00)',
      'calm-sadness': 'linear-gradient(-45deg, #00FF71, #0065FF)',  
      'calm-love': 'linear-gradient(-45deg, #00FF71, #FF8200)',  
      'calm-energetic': 'linear-gradient(-45deg, #00FF71, #FF8200)',
      'love-happiness': 'linear-gradient(-45deg, #F100FF, #FFFA00)',
      'love-sadness': 'linear-gradient(-45deg, #F100FF, #0065FF)',
      'love-calm': 'linear-gradient(-45deg, #F100FF, #00FF71)',
      'love-energetic': 'linear-gradient(-45deg, #F100FF, #FF8200)',
      'energetic-happiness': 'linear-gradient(-45deg, #FF8200, #FFFA00)',
      'energetic-sadness': 'linear-gradient(-45deg, #FF8200, #0065FF)',
      'energetic-love': 'linear-gradient(-45deg, #FF8200, #F100FF)',
      'energetic-calm': 'linear-gradient(-45deg, #FF8200, #00FF71)',               

    };

    return gradients[combination] ||  'linear-gradient(-45deg, #242424, #242424)';
  };


  return (
<div
  className={`${animateGradient ? 'animate-gradient' : 'gradient-container'}`}
  style={{
    ...(animateGradient && {
      background: backgroundStyle,
      animation: 'animate-gradient 6s ease infinite alternate',
      backgroundSize: '300%',
    }),
  }}
>      <div className="flex rounded-sm justify-center items-center">
        <Navbar className='flex justify-center items-center' maxWidth='full' classNames={{
          wrapper: "bg-[#242424]"
        }} position="sticky">
          <NavbarBrand className="flex justify-center">
            <Link to="." onClick={handleRefresh}>
              <p className={`font-extrabold text-white text-2xl `}>HYOU</p>
            </Link>
          </NavbarBrand>
        </Navbar>
      </div>

      <div className={`main flex flex-col justify-center items-center h-screen mt-[-5vw]`}>
        {showSearch && (
          <>
            <p className={`text-[8vw] font-extrabold bg-gradient-to-r from-blue-400 to-pink-600 text-transparent bg-clip-text animate-gradient ${showResults ? 'hidden' : ''} ${fade ? 'fade-in' : 'fade-out'}`}>HYOU</p>
            <p className={`font-bold text-xl text-white mb-12 ${showResults ? 'hidden' : ''} ${fade ? 'fade-in' : 'fade-out'}`}>Find your mood</p>
          </>
        )}
        {showSearch && (
          <div className={`${fade ? 'fade-in' : 'fade-out'}`}>
            <form className='flex flex-col justify-center items-center' onSubmit={handleSearch}>
              <div className='w-[30vw]'>
                <Input
                  label='Search a Song'
                  radius='xl'
                  isClearable
                  classNames={{
                    input: [
                      'text-black/90 dark:text-white/90',
                      'placeholder:text-default-700/50 dark:placeholder:text-white/60',
                    ],
                    innerWrapper: 'bg-transparent',
                    inputWrapper: [
                      'shadow-xl',
                      'bg-default-200/50',
                      'backdrop-blur-xl',
                      'backdrop-saturate-200',
                      'hover:bg-default-200/70',
                      'group-data-[focused=true]:bg-default-200/50',
                      '!cursor-text',
                    ],
                  }}
                  value={query}
                  onChange={handleChange}
                />
              </div>
              <Button variant='flat' radius="xl" className="w-[10vw] mt-6 bg-gradient-to-tr from-blue-400 to-pink-600 text-white shadow-lg" type='submit' onClick={handleFadeOut}>
                Search
              </Button>
            </form>
          </div>
        )}
        {showResults && (
          <div className={`emotions ${fadeEmotions ? 'fade-in' : ''}`}>
            <p className={`text-white text-2xl ${fadeEmotions ? 'fade-in' : 'hidden'}`}>{emotions}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
