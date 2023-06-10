import { useState } from 'react'
import {navbar} from "@nextui-org/theme";
import './App.css';
import { Navbar, NavbarContent, NavbarItem, NavbarBrand } from '@nextui-org/react';
import { Input, Button } from '@nextui-org/react';
import { Link } from 'react-router-dom';
import { color } from 'framer-motion';
import axios from 'axios';

function App() {
  const [query, searchQuery] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('/predict-emotion', { track_name: searchQuery });
      // Handle the response from the backend
      console.log(response.data);
    } catch (error) {
      // Handle errors
      console.error(error);
    }
  };

  const handleChange = (e) => {
    searchQuery(e.target.value);
  };

  return (
    <body className='flex flex-col'>
      <div className="flex rounded-sm justify-center items-center">
        <Navbar className='flex justify-center items-center' maxWidth='full' classNames={{
    wrapper: "bg-[#242424]"}} position = "sticky">
          <NavbarBrand className="flex justify-center"> 
            <p className="font-extrabold text-white text-2xl">HYOU</p>
          </NavbarBrand>
        </Navbar>
      </div>

      <div className='main flex flex-col justify-center items-center h-screen mt-[-5vw]'>
          <p className=" text-[8vw] font-extrabold bg-gradient-to-r from-blue-400 to-pink-600 text-transparent bg-clip-text animate-gradient">HYOU</p>
          <p className='font-bold text-xl text-white mb-12'> Find your mood </p>
          <div>
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
                onChange = {handleChange}
              />
            </div>
            <Button variant='flat' radius="xl" className="w-[10vw] mt-6 bg-gradient-to-tr from-blue-400 to-pink-600 text-white shadow-lg" type='submit'>
              Search
            </Button>
          </form>
        </div>
      </div>
    </body>
  )
}

export default App
