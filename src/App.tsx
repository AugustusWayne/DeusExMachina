import { useState, use } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import EEGAttentionPlatform from './EEGAttentionPlatform'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div className="dark">
        <EEGAttentionPlatform />
      </div>
    </>
  )
}

export default App
