import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Upload, Brain, Activity, X } from 'lucide-react';

const EEGAttentionPlatform = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [attentionScore, setAttentionScore] = useState(0);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.substr(0, 5) === "image") {
      setSelectedImage(URL.createObjectURL(file));
      setAnalysisComplete(false);
    } else {
      alert("Please select an image file");
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.substr(0, 5) === "image") {
      setSelectedImage(URL.createObjectURL(file));
      setAnalysisComplete(false);
    } else {
      alert("Please drop an image file");
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
    setAnalysisComplete(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const analyzeAttention = () => {
    // Simulating analysis with random data
    setTimeout(() => {
      setAttentionScore(Math.floor(Math.random() * 100));
      setAnalysisComplete(true);
    }, 2000);
  };

  const getRandomData = (count) => {
    return Array.from({ length: count }, (_, i) => ({
      name: `T${i + 1}`,
      value: Math.floor(Math.random() * 100)
    }));
  };

  const brainwaveData = [
    { name: 'Delta', value: Math.random() * 100 },
    { name: 'Theta', value: Math.random() * 100 },
    { name: 'Alpha', value: Math.random() * 100 },
    { name: 'Beta', value: Math.random() * 100 },
    { name: 'Gamma', value: Math.random() * 100 },
  ];
  const timeSeriesData = getRandomData(10);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Left side - Image Input */}
      <div className="w-1/2 p-4 flex flex-col">
        <Card className="flex-grow bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-white">EEG Image Analysis</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center h-full">
            <div 
              className="w-full h-64 border-2 border-dashed border-gray-600 rounded-lg flex flex-col items-center justify-center cursor-pointer relative"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              {selectedImage ? (
                <>
                  <img src={selectedImage} alt="EEG" className="max-w-full max-h-full rounded-lg" />
                  <button 
                    onClick={(e) => { e.stopPropagation(); removeImage(); }} 
                    className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
                  >
                    <X size={20} />
                  </button>
                </>
              ) : (
                <>
                  <Upload size={48} className="text-gray-400 mb-4" />
                  <p className="text-gray-300">Click or drag and drop to upload an image</p>
                </>
              )}
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              ref={fileInputRef}
            />
            <Button 
              onClick={analyzeAttention} 
              disabled={!selectedImage || analysisComplete} 
              className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold"
            >
              Analyze Attention
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Right side - Dashboard */}
      <div className="w-1/2 p-4 flex flex-col">
        <Card className="mb-4 bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-xl font-semibold flex items-center text-white">
              <Brain className="mr-2" /> Attention Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={attentionScore} className="w-full h-4 mb-2" />
            <p className="text-center mt-2 text-3xl font-bold text-white">{attentionScore}%</p>
          </CardContent>
        </Card>

        <Tabs defaultValue="brainwaves" className="flex-grow">
          <TabsList className="grid w-full grid-cols-2 bg-gray-800">
            <TabsTrigger value="brainwaves" className="text-gray-300 data-[state=active]:text-white">Brainwave Activity</TabsTrigger>
            <TabsTrigger value="timeseries" className="text-gray-300 data-[state=active]:text-white">Attention Over Time</TabsTrigger>
          </TabsList>
          <TabsContent value="brainwaves">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-xl font-semibold flex items-center text-white">
                  <Activity className="mr-2" /> Brainwave Activity
                </CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={brainwaveData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4b5563" />
                    <XAxis dataKey="name" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '4px', boxShadow: '0 2px 5px rgba(0,0,0,0.2)' }}
                      itemStyle={{ color: '#e5e7eb' }}
                      labelStyle={{ color: '#e5e7eb' }}
                    />
                    <Bar dataKey="value" fill="#60a5fa" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="timeseries">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-xl font-semibold flex items-center text-white">
                  <Activity className="mr-2" /> Attention Over Time
                </CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4b5563" />
                    <XAxis dataKey="name" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '4px', boxShadow: '0 2px 5px rgba(0,0,0,0.2)' }}
                      itemStyle={{ color: '#e5e7eb' }}
                      labelStyle={{ color: '#e5e7eb' }}
                    />
                    <Line type="monotone" dataKey="value" stroke="#34d399" strokeWidth={2} dot={{ fill: '#34d399', strokeWidth: 2 }} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default EEGAttentionPlatform;