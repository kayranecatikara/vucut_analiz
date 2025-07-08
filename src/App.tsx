import React, { useEffect, useState, useRef } from 'react';
import { Camera, Activity, Users, Zap, AlertCircle } from 'lucide-react';
import { io } from 'socket.io-client';

function App() {
  const [socket, setSocket] = useState(null);
  const [analysis, setAnalysis] = useState({
    omuz_genisligi: 0,
    bel_genisligi: 0,
    omuz_bel_orani: 0,
    vucut_tipi: 'Analiz Bekleniyor',
    mesafe: 0,
    confidence: 0
  });
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    message: 'Bağlantı bekleniyor...',
    timestamp: Date.now()
  });
  const [isStreaming, setIsStreaming] = useState(false);
  const [frameRate, setFrameRate] = useState(0);
  const [lastFrameTime, setLastFrameTime] = useState(Date.now());
  const [frameCount, setFrameCount] = useState(0);

  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // WebSocket bağlantısı ve otomatik yeniden bağlanma
  const connectWebSocket = () => {
    try {
      const ws = io(`http://localhost:5000`, {
        transports: ['websocket'],
        autoConnect: true
      });
      
      ws.on('connect', () => {
        console.log('✅ WebSocket bağlantısı kuruldu');
        setConnectionStatus({
          connected: true,
          message: 'Bağlantı başarılı',
          timestamp: Date.now()
        });
        setSocket(ws);
      });

      ws.on('video_frame', (data) => {
        updateVideoFrame(data.frame || data);
        updateFrameRate();
      });

      ws.on('analyze_result', (data) => {
        updateAnalysisData(data.data || data);
      });

      ws.on('stream_started', () => {
        setIsStreaming(true);
      });

      ws.on('stream_stopped', () => {
        setIsStreaming(false);
      });

      ws.on('disconnect', () => {
        console.log('❌ WebSocket bağlantısı kesildi');
        setConnectionStatus({
          connected: false,
          message: 'Bağlantı kesildi - Yeniden bağlanıyor...',
          timestamp: Date.now()
        });
        setSocket(null);
        setIsStreaming(false);
        
        // 3 saniye sonra otomatik yeniden bağlan
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 3000);
      });

      ws.on('error', (error) => {
        console.error('WebSocket hatası:', error);
        setConnectionStatus({
          connected: false,
          message: 'Bağlantı hatası',
          timestamp: Date.now()
        });
      });

    } catch (error) {
      console.error('WebSocket bağlantısı kurulamadı:', error);
      setConnectionStatus({
        connected: false,
        message: 'Bağlantı kurulamadı',
        timestamp: Date.now()
      });
    }
  };

  const updateVideoFrame = (frameData) => {
    if (imageRef.current) {
      imageRef.current.src = `data:image/jpeg;base64,${frameData}`;
    }
  };

  const updateFrameRate = () => {
    const now = Date.now();
    const timeDiff = now - lastFrameTime;
    
    if (timeDiff > 1000) { // Her saniye güncelle
      setFrameRate(Math.round(frameCount * 1000 / timeDiff));
      setFrameCount(0);
      setLastFrameTime(now);
    } else {
      setFrameCount(prev => prev + 1);
    }
  };

  const updateAnalysisData = (data) => {
    console.log('Gelen analiz verisi:', data); // Debug için
    setAnalysis({
      omuz_genisligi: parseFloat(data.omuz_genisligi) || 0,
      bel_genisligi: parseFloat(data.bel_genisligi) || 0,
      omuz_bel_orani: parseFloat(data.omuz_bel_orani) || 0,
      vucut_tipi: data.vucut_tipi || 'Analiz Bekleniyor',
      mesafe: parseFloat(data.mesafe) || 0,
      confidence: data.confidence || 0
    });
  };

  const sendWebSocketMessage = (type, data) => {
    if (socket && socket.connected) {
      socket.emit(type, data);
    }
  };

  const startVideo = () => {
    if (connectionStatus.connected && !isStreaming) {
      sendWebSocketMessage('start_video');
    }
  };

  const stopVideo = () => {
    if (connectionStatus.connected && isStreaming) {
      sendWebSocketMessage('stop_video');
    }
  };

  const getBodyTypeColor = (type) => {
    switch (type.toLowerCase()) {
      case 'ektomorf': return '#3B82F6';
      case 'mezomorf': return '#10B981';
      case 'endomorf': return '#F59E0B';
      default: return '#6B7280';
    }
  };

  const getBodyTypeDescription = (type) => {
    switch (type.toLowerCase()) {
      case 'ektomorf': return 'İnce yapılı, hızlı metabolizma';
      case 'mezomorf': return 'Atletik yapı, orta metabolizma';
      case 'endomorf': return 'Geniş yapılı, yavaş metabolizma';
      default: return 'Analiz devam ediyor...';
    }
  };

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Başlık */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Activity className="h-8 w-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-slate-900">
                Canlı Vücut Analizi
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${connectionStatus.connected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-slate-600">
                  {connectionStatus.message}
                </span>
              </div>
              {isStreaming && (
                <div className="flex items-center space-x-2 text-sm text-slate-600">
                  <Zap className="h-4 w-4" />
                  <span>{frameRate} FPS</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Ana İçerik */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Video Akışı */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-slate-900 flex items-center">
                    <Camera className="h-5 w-5 mr-2 text-blue-600" />
                    Kamera Görüntüsü
                  </h2>
                  <div className="flex space-x-3">
                    <button
                      onClick={startVideo}
                      disabled={!connectionStatus.connected || isStreaming}
                      className={`
                        px-4 py-2 rounded-lg font-medium transition-all duration-200
                        ${!connectionStatus.connected || isStreaming
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-700 text-white shadow-sm hover:shadow-md'
                        }
                      `}
                    >
                      Başlat
                    </button>
                    <button
                      onClick={stopVideo}
                      disabled={!connectionStatus.connected || !isStreaming}
                      className={`
                        px-4 py-2 rounded-lg font-medium transition-all duration-200
                        ${!connectionStatus.connected || !isStreaming
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                          : 'bg-red-600 hover:bg-red-700 text-white shadow-sm hover:shadow-md'
                        }
                      `}
                    >
                      Durdur
                    </button>
                  </div>
                </div>
                
                <div className="relative">
                  {isStreaming ? (
                    <div className="relative bg-black rounded-lg overflow-hidden max-w-full">
                      <img
                        ref={imageRef}
                        alt="Kamera Görüntüsü"
                        className="w-full h-auto max-h-[500px] object-contain"
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full pointer-events-none"
                      />
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                        Sol: RGB + Pose Detection | Sağ: Derinlik Görüntüsü
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-100 rounded-lg h-[400px] flex items-center justify-center">
                      <div className="text-center">
                        <Camera className="h-16 w-16 text-slate-400 mx-auto mb-4" />
                        <p className="text-slate-600 text-lg">
                          Kamera görüntüsü bekleniyor...
                        </p>
                        <p className="text-slate-500 text-sm mt-2">
                          RGB + Derinlik görüntüsü için "Başlat" butonuna tıklayın
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Analiz Paneli */}
          <div className="space-y-6">
            
            {/* Vücut Ölçüleri */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                <Users className="h-5 w-5 mr-2 text-blue-600" />
                Vücut Ölçüleri
              </h3>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                  <span className="text-slate-600">Omuz Genişliği</span>
                  <span className="font-semibold text-slate-900">
                    {analysis.omuz_genisligi > 0 ? `${Number(analysis.omuz_genisligi).toFixed(1)} cm` : '—'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                  <span className="text-slate-600">Bel Genişliği</span>
                  <span className="font-semibold text-slate-900">
                    {analysis.bel_genisligi > 0 ? `${Number(analysis.bel_genisligi).toFixed(1)} cm` : '—'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                  <span className="text-slate-600">Omuz/Bel Oranı</span>
                  <span className="font-semibold text-slate-900">
                    {analysis.omuz_bel_orani > 0 ? Number(analysis.omuz_bel_orani).toFixed(2) : '—'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                  <span className="text-slate-600">Mesafe</span>
                  <span className="font-semibold text-slate-900">
                    {analysis.mesafe > 0 ? `${Number(analysis.mesafe).toFixed(1)} m` : '—'}
                  </span>
                </div>
              </div>
            </div>

            {/* Vücut Tipi Analizi */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">
                Vücut Tipi Analizi
              </h3>
              
              <div className="text-center">
                <div 
                  className="w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center text-white font-bold text-lg"
                  style={{ backgroundColor: getBodyTypeColor(analysis.vucut_tipi) }}
                >
                  {analysis.vucut_tipi.charAt(0).toUpperCase()}
                </div>
                
                <h4 className="text-xl font-semibold text-slate-900 mb-2">
                  {analysis.vucut_tipi}
                </h4>
                
                <p className="text-slate-600 text-sm mb-4">
                  {getBodyTypeDescription(analysis.vucut_tipi)}
                </p>
                
                {analysis.confidence > 0 && (
                  <div className="mt-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-slate-600">Güvenilirlik</span>
                      <span className="text-sm font-medium text-slate-900">
                        {(analysis.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${analysis.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Sistem Durumu */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">
                Sistem Durumu
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Kamera</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-500' : 'bg-slate-300'}`} />
                    <span className="text-sm text-slate-900">
                      {isStreaming ? 'Aktif' : 'Pasif'}
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Bağlantı</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${connectionStatus.connected ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-sm text-slate-900">
                      {connectionStatus.connected ? 'Bağlı' : 'Bağlı Değil'}
                    </span>
                  </div>
                </div>
                
                {isStreaming && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-600">FPS</span>
                    <span className="text-sm text-slate-900">{frameRate}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;