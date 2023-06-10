import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { NextUIProvider } from '@nextui-org/react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <NextUIProvider>
    <React.StrictMode>
      <Router>
        <Routes>
          <Route path = "/" element = {<App/>} />
          <Route path = "/about" />
        </Routes>
      </Router>
    </React.StrictMode>
  </NextUIProvider>
)
