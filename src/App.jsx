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
      setEmotions(response.data.emotion.join(' '));

      setShowResults(true);
      setShowSearch(false);
      setAnimateGradient(true);

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

  return (
    <div className={`gradient-container ${animateGradient ? 'animate-gradient' : ''}`}>
      <div className="flex rounded-sm justify-center items-center">
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
