import React, { useEffect, useState, useRef } from 'react';
import { Camera, Activity, Users, Zap, AlertCircle, Play, Square, Clock, Target, Utensils, Calendar, ChefHat, Calculator, Image } from 'lucide-react';
import { io } from 'socket.io-client';

function App() {
  const [socket, setSocket] = useState(null);
  const [testResults, setTestResults] = useState(null);
  
  // Kalori hesaplama state'leri
  const [calorieCalculationStatus, setCalorieCalculationStatus] = useState({
    active: false,
    counting: false,
    capturing: false,
    analyzing: false,
    completed: false,
    countdown: 0
  });
  const [capturedFoodImage, setCapturedFoodImage] = useState(null);
  const [foodAnalysisResults, setFoodAnalysisResults] = useState(null);
  
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    message: 'Bağlantı bekleniyor...',
    timestamp: Date.now()
  });
  const [testStatus, setTestStatus] = useState({
    running: false,
    timeLeft: 0,
    completed: false
  });

  const imageRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // WebSocket bağlantısı ve otomatik yeniden bağlanma
  const connectWebSocket = () => {
    try {
      // Backend URL'ini düzelt
      const backendUrl = 'http://localhost:5000';
      const ws = io(backendUrl, {
        transports: ['websocket'],
        autoConnect: true,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5,
        timeout: 10000,
        forceNew: false
      });
      
      ws.on('connect', () => {
        console.log('✅ WebSocket bağlantısı kuruldu:', backendUrl);
        setConnectionStatus({
          connected: true,
          message: 'Bağlantı başarılı',
          timestamp: Date.now()
        });
        setSocket(ws);
        
        // Bağlantı test et
        ws.emit('ping', { timestamp: Date.now() });
      });

      // Heartbeat sistemi
      const heartbeat = setInterval(() => {
        if (ws.connected) {
          ws.emit('ping', { timestamp: Date.now() });
        }
      }, 30000); // 30 saniyede bir ping gönder

      ws.on('pong', (data) => {
        console.log('💓 Pong alındı, bağlantı sağlıklı');
      });

      ws.on('heartbeat', (data) => {
        console.log('💓 Heartbeat alındı');
      });

      ws.on('connection_ok', (data) => {
        console.log('✅ Bağlantı durumu: OK');
        setConnectionStatus({
          connected: true,
          message: 'Bağlantı onaylandı',
          timestamp: Date.now()
        });
      });

      ws.on('test_frame', (data) => {
        updateVideoFrame(data.frame);
        setTestStatus(prev => ({
          ...prev,
          timeLeft: data.time_left || 0
        }));
      });

      ws.on('test_started', (data) => {
        setTestStatus({
          running: true,
          timeLeft: data.duration || 10,
          completed: false
        });
        setTestResults(null);
      });

      ws.on('test_completed', (data) => {
        setTestStatus({
          running: false,
          timeLeft: 0,
          completed: true
        });
        setTestResults(data);
        
        // Test tamamlandıktan sonra bağlantıyı koru
        setTimeout(() => {
          ws.emit('check_connection');
        }, 2000);
        
        setConnectionStatus({
          connected: true,
          message: 'Test tamamlandı - Hazır',
          timestamp: Date.now()
        });
        
        console.log('Test tamamlandı:', data);
      });

      // Kalori hesaplama olayları
      ws.on('food_capture_countdown', (data) => {
        setCalorieCalculationStatus(prev => ({
          ...prev,
          counting: true,
          countdown: data.count
        }));
      });

      ws.on('food_capture_started', () => {
        setCalorieCalculationStatus(prev => ({
          ...prev,
          counting: false,
          capturing: true
        }));
      });

      ws.on('food_analysis_started', () => {
        setCalorieCalculationStatus(prev => ({
          ...prev,
          capturing: false,
          analyzing: true
        }));
      });

      ws.on('food_analysis_result', (data) => {
        setCapturedFoodImage(data.image);
        setFoodAnalysisResults(data.analysis);
        setCalorieCalculationStatus({
          active: false,
          counting: false,
          capturing: false,
          analyzing: false,
          completed: true,
          countdown: 0
        });
        console.log('Yemek analizi tamamlandı:', data);
      });

      ws.on('food_analysis_error', (data) => {
        setCalorieCalculationStatus({
          active: false,
          counting: false,
          capturing: false,
          analyzing: false,
          completed: false,
          countdown: 0
        });
        setConnectionStatus({
          connected: true,
          message: `Kalori hesaplama hatası: ${data.message}`,
          timestamp: Date.now()
        });
      });

      ws.on('test_stopped', () => {
        setTestStatus({
          running: false,
          timeLeft: 0,
          completed: false
        });
        // Kalori hesaplama da durdur
        setCalorieCalculationStatus({
          active: false,
          counting: false,
          capturing: false,
          analyzing: false,
          completed: false,
          countdown: 0
        });
      });

      ws.on('test_error', (error) => {
        console.error('Test hatası:', error);
        setTestStatus({
          running: false,
          timeLeft: 0,
          completed: false
        });
        setConnectionStatus({
          connected: true,
          message: `Test hatası: ${error}`,
          timestamp: Date.now()
        });
      });

      ws.on('disconnect', () => {
        console.log('❌ WebSocket bağlantısı kesildi');
        clearInterval(heartbeat);
        setConnectionStatus({
          connected: false,
          message: 'Bağlantı kesildi',
          timestamp: Date.now()
        });
        setSocket(null);
        setTestStatus({
          running: false,
          timeLeft: 0,
          completed: false
        });
      });

      ws.on('reconnect', (attemptNumber) => {
        console.log(`✅ Yeniden bağlandı (deneme ${attemptNumber})`);
        setConnectionStatus({
          connected: true,
          message: 'Yeniden bağlandı',
          timestamp: Date.now()
        });
      });

      ws.on('reconnect_attempt', (attemptNumber) => {
        console.log(`🔄 Yeniden bağlanma denemesi ${attemptNumber}`);
        setConnectionStatus({
          connected: false,
          message: `Yeniden bağlanıyor... (${attemptNumber})`,
          timestamp: Date.now()
        });
      });

      ws.on('reconnect_failed', () => {
        console.log('❌ Yeniden bağlanma başarısız');
        setConnectionStatus({
          connected: false,
          message: 'Bağlantı başarısız - Sunucuyu kontrol edin',
          timestamp: Date.now()
        });
      });

      ws.on('error', (error) => {
        console.error('WebSocket hatası:', error);
        setConnectionStatus({
          connected: false,
          message: `Bağlantı hatası: ${error}`,
          timestamp: Date.now()
        });
      });

    } catch (error) {
      console.error('WebSocket bağlantısı kurulamadı:', error);
      setConnectionStatus({
        connected: false,
        message: `Bağlantı kurulamadı: ${error}`,
        timestamp: Date.now()
      });
    }
  };

  const updateVideoFrame = (frameData) => {
    if (imageRef.current) {
      imageRef.current.src = `data:image/jpeg;base64,${frameData}`;
    }
  };

  const sendWebSocketMessage = (type, data) => {
    if (socket && socket.connected) {
      socket.emit(type, data);
    }
  };

  const startTest = () => {
    if (connectionStatus.connected && !testStatus.running && socket) {
      sendWebSocketMessage('start_test');
      console.log('Test başlatma komutu gönderildi');
    }
  };

  const stopTest = () => {
    if (connectionStatus.connected && testStatus.running && socket) {
      sendWebSocketMessage('stop_test');
      console.log('Test durdurma komutu gönderildi');
    }
  };

  const startCalorieCalculation = () => {
    if (connectionStatus.connected && !testStatus.running && !calorieCalculationStatus.active && socket) {
      setCalorieCalculationStatus({
        active: true,
        counting: false,
        capturing: false,
        analyzing: false,
        completed: false,
        countdown: 0
      });
      setCapturedFoodImage(null);
      setFoodAnalysisResults(null);
      sendWebSocketMessage('take_food_photo');
      console.log('Kalori hesaplama başlatıldı');
    }
  };

  const getBodyTypeColor = (type) => {
    switch (type?.toLowerCase()) {
      case 'ektomorf': return '#3B82F6';
      case 'mezomorf': return '#10B981';
      case 'endomorf': return '#F59E0B';
      default: return '#6B7280';
    }
  };

  const getBodyTypeDescription = (type) => {
    switch (type?.toLowerCase()) {
      case 'ektomorf': return 'İnce yapılı, hızlı metabolizma';
      case 'mezomorf': return 'Atletik yapı, orta metabolizma';
      case 'endomorf': return 'Geniş yapılı, yavaş metabolizma';
      default: return 'Analiz devam ediyor...';
    }
  };

  // Haftalık yemek planı oluşturma
  const createWeeklyMealPlan = (dietData) => {
    if (!dietData || !dietData.ogun_plani) {
      console.log('Diet data veya ogun_plani bulunamadı:', dietData);
      return null;
    }

    const days = ['pazartesi', 'sali', 'carsamba', 'persembe', 'cuma', 'cumartesi', 'pazar'];
    const dayNames = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar'];
    const meals = [
      { key: 'kahvalti', name: 'Kahvaltı', icon: '🌅' },
      { key: 'ara_ogun_1', name: 'Ara Öğün 1', icon: '🍎' },
      { key: 'ogle', name: 'Öğle', icon: '☀️' },
      { key: 'ara_ogun_2', name: 'Ara Öğün 2', icon: '🥜' },
      { key: 'aksam', name: 'Akşam', icon: '🌙' },
      { key: 'gece', name: 'Gece', icon: '🌃' }
    ];

    console.log('Ogun plani:', dietData.ogun_plani);
    return { days, dayNames, meals, plan: dietData.ogun_plani };
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

  const weeklyPlan = testResults?.diyet_onerileri ? createWeeklyMealPlan(testResults.diyet_onerileri) : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Başlık */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Activity className="h-8 w-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-slate-900">
                Vücut Analizi ve Diyet Önerisi
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${connectionStatus.connected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-slate-600">
                  {connectionStatus.message}
                </span>
              </div>
              {testStatus.running && (
                <div className="flex items-center space-x-2 text-sm text-slate-600">
                  <Clock className="h-4 w-4" />
                  <span>{testStatus.timeLeft}s</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Ana İçerik */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Test Alanı - Sol Taraf */}
          <div className="lg:col-span-2 space-y-6">
            {/* Kamera Test Alanı */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-slate-900 flex items-center">
                    <Target className="h-5 w-5 mr-2 text-blue-600" />
                    Vücut Analizi Testi
                  </h2>
                  <div className="flex space-x-3">
                    <button
                      onClick={startTest}
                      disabled={!connectionStatus.connected || testStatus.running}
                      className={`
                        px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2
                        ${!connectionStatus.connected || testStatus.running
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                          : 'bg-green-600 hover:bg-green-700 text-white shadow-sm hover:shadow-md'
                        }
                      `}
                    >
                      <Play className="h-4 w-4" />
                      <span>Teste Başla</span>
                    </button>
                    <button
                      onClick={stopTest}
                      disabled={!connectionStatus.connected || !testStatus.running}
                      className={`
                        px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2
                        ${!connectionStatus.connected || !testStatus.running
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                          : 'bg-red-600 hover:bg-red-700 text-white shadow-sm hover:shadow-md'
                        }
                      `}
                    >
                      <Square className="h-4 w-4" />
                      <span>Durdur</span>
                    </button>
                  </div>
                </div>
                
                {/* Kalori Hesaplama Butonu */}
                <div className="mb-4 p-4 bg-orange-50 rounded-lg border border-orange-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-orange-900 mb-1 flex items-center">
                        <Calculator className="h-4 w-4 mr-2" />
                        Kalori Hesaplama
                      </h3>
                      <p className="text-orange-800 text-sm">
                        Yemek fotoğrafı çekerek kalori hesaplayın
                      </p>
                    </div>
                    <button
                      onClick={startCalorieCalculation}
                      disabled={!connectionStatus.connected || testStatus.running || calorieCalculationStatus.active}
                      className={`
                        px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2
                        ${!connectionStatus.connected || testStatus.running || calorieCalculationStatus.active
                          ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                          : 'bg-orange-600 hover:bg-orange-700 text-white shadow-sm hover:shadow-md'
                        }
                      `}
                    >
                      <Image className="h-4 w-4" />
                      <span>
                        {calorieCalculationStatus.active ? 'İşleniyor...' : 'Fotoğraf Çek'}
                      </span>
                    </button>
                  </div>
                </div>
                
                {/* Test Açıklaması */}
                {!testStatus.running && !testStatus.completed && (
                  <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <h3 className="font-semibold text-blue-900 mb-2">Test Nasıl Çalışır?</h3>
                    <ul className="text-blue-800 text-sm space-y-1">
                      <li>• Kameraya 1-2 metre mesafede durun</li>
                      <li>• Kollarınızı yana açın</li>
                      <li>• 10 saniye boyunca sabit durun</li>
                      <li>• Sistem vücut tipinizi analiz edecek</li>
                      <li>• Size özel diyet önerileri sunulacak</li>
                    </ul>
                  </div>
                )}
                
                <div className="relative">
                  {testStatus.running ? (
                    <div className="relative bg-black rounded-lg overflow-hidden max-w-full">
                      <img
                        ref={imageRef}
                        alt="Test Görüntüsü"
                        className="w-full h-auto max-h-[500px] object-contain"
                      />
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                        Sol: RGB + Pose Detection | Sağ: Derinlik Görüntüsü
                      </div>
                      {testStatus.timeLeft > 0 && (
                        <div className="absolute top-2 right-2 bg-red-600 text-white px-3 py-1 rounded-full font-bold">
                          {testStatus.timeLeft}s
                        </div>
                      )}
                    </div>
                  ) : calorieCalculationStatus.active ? (
                    <div className="bg-slate-100 rounded-lg h-[400px] flex items-center justify-center">
                      <div className="text-center">
                        {calorieCalculationStatus.counting && (
                          <>
                            <div className="text-6xl font-bold text-orange-600 mb-4">
                              {calorieCalculationStatus.countdown}
                            </div>
                            <p className="text-slate-600 text-lg">Fotoğraf çekiliyor...</p>
                          </>
                        )}
                        {calorieCalculationStatus.capturing && (
                          <>
                            <Camera className="h-16 w-16 text-orange-600 mx-auto mb-4 animate-pulse" />
                            <p className="text-slate-600 text-lg">Fotoğraf çekiliyor...</p>
                          </>
                        )}
                        {calorieCalculationStatus.analyzing && (
                          <>
                            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-orange-600 mx-auto mb-4"></div>
                            <p className="text-slate-600 text-lg">Yemek analiz ediliyor...</p>
                          </>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-slate-100 rounded-lg h-[400px] flex items-center justify-center">
                      <div className="text-center">
                        <Camera className="h-16 w-16 text-slate-400 mx-auto mb-4" />
                        <p className="text-slate-600 text-lg">
                          {testStatus.completed ? 'Test tamamlandı!' : 'Test başlatmak için "Teste Başla" butonuna tıklayın'}
                        </p>
                        <p className="text-slate-500 text-sm mt-2">
                          {testStatus.completed ? 'Sonuçlarınızı sağ panelden inceleyebilirsiniz' : '10 saniye sürecek analiz için hazır olun'}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Haftalık Yemek Planı Tablosu */}
            {weeklyPlan && (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="text-xl font-semibold text-slate-900 mb-6 flex items-center">
                  <Calendar className="h-5 w-5 mr-2 text-green-600" />
                  Haftalık Yemek Planı - {testResults.vucut_tipi}
                </h3>
                
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-slate-50">
                        <th className="border border-slate-200 px-3 py-2 text-left font-semibold text-slate-700">
                          Öğün
                        </th>
                        {weeklyPlan.days.map(day => (
                          <th key={day} className="border border-slate-200 px-3 py-2 text-center font-semibold text-slate-700 min-w-[120px]">
                            {weeklyPlan.dayNames[weeklyPlan.days.indexOf(day)]}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {weeklyPlan.meals.map(meal => (
                        <tr key={meal.key} className="hover:bg-slate-50">
                          <td className="border border-slate-200 px-3 py-3 font-medium text-slate-800 bg-slate-50">
                            <div className="flex items-center space-x-2">
                              <span className="text-lg">{meal.icon}</span>
                              <span>{meal.name}</span>
                            </div>
                          </td>
                          {weeklyPlan.days.map(day => (
                            <td key={`${meal.key}-${day}`} className="border border-slate-200 px-3 py-3 text-sm text-slate-600">
                              {weeklyPlan.plan[day] && weeklyPlan.plan[day][meal.key] ? weeklyPlan.plan[day][meal.key] : '—'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Ek Bilgiler */}
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                    <h4 className="font-semibold text-green-800 mb-2 flex items-center">
                      <ChefHat className="h-4 w-4 mr-2" />
                      Beslenme İpuçları
                    </h4>
                    <ul className="text-green-700 text-sm space-y-1">
                      <li>• Öğünleri düzenli saatlerde tüketin</li>
                      <li>• Bol su için (günde 2-3 litre)</li>
                      <li>• Porsiyonları kontrol edin</li>
                      <li>• Yavaş yemek yiyin</li>
                    </ul>
                  </div>
                  
                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <h4 className="font-semibold text-blue-800 mb-2">Egzersiz Önerileri</h4>
                    <ul className="text-blue-700 text-sm space-y-1">
                      {testResults.vucut_tipi === 'Ektomorf' && (
                        <>
                          <li>• Ağırlık antrenmanı (3-4 gün/hafta)</li>
                          <li>• Kısa süreli kardio</li>
                          <li>• Compound hareketler</li>
                        </>
                      )}
                      {testResults.vucut_tipi === 'Mezomorf' && (
                        <>
                          <li>• Karma antrenman programı</li>
                          <li>• Orta süreli kardio</li>
                          <li>• Çeşitli spor aktiviteleri</li>
                        </>
                      )}
                      {testResults.vucut_tipi === 'Endomorf' && (
                        <>
                          <li>• Yoğun kardio (5-6 gün/hafta)</li>
                          <li>• Yüksek tekrarlı ağırlık</li>
                          <li>• Aktif yaşam tarzı</li>
                        </>
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sonuçlar Paneli - Sağ Taraf */}
          <div className="space-y-6">
            
            {/* Kalori Hesaplama Sonuçları */}
            {calorieCalculationStatus.completed && foodAnalysisResults && (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                  <Calculator className="h-5 w-5 mr-2 text-orange-600" />
                  Kalori Analizi
                </h3>
                
                {/* Çekilen Fotoğraf */}
                {capturedFoodImage && (
                  <div className="mb-4">
                    <img 
                      src={`data:image/jpeg;base64,${capturedFoodImage}`}
                      alt="Çekilen Yemek"
                      className="w-full h-48 object-cover rounded-lg border border-slate-200"
                    />
                  </div>
                )}
                
                {/* Tespit Edilen Yemekler */}
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-orange-50 rounded-lg">
                    <span className="font-semibold text-orange-900">Toplam Kalori</span>
                    <span className="text-xl font-bold text-orange-600">
                      {foodAnalysisResults.total_calories} kcal
                    </span>
                  </div>
                  
                  {foodAnalysisResults.detected_foods && foodAnalysisResults.detected_foods.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-slate-800 mb-2">Tespit Edilen Yemekler</h4>
                      <div className="space-y-2">
                        {foodAnalysisResults.detected_foods.map((food, index) => (
                          <div key={index} className="flex justify-between items-center p-2 bg-slate-50 rounded">
                            <span className="text-slate-700">{food.name}</span>
                            <span className="font-medium text-slate-900">{food.calories} kcal</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {foodAnalysisResults.confidence && (
                    <div className="mt-3">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-slate-600">Güvenilirlik</span>
                        <span className="text-sm font-medium text-slate-900">
                          {(foodAnalysisResults.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div 
                          className="bg-orange-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${foodAnalysisResults.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* Test Sonuçları */}
            {testResults && (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                  <Users className="h-5 w-5 mr-2 text-blue-600" />
                  Test Sonuçları
                </h3>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-slate-600">Omuz Genişliği</span>
                    <span className="font-semibold text-slate-900">
                      {testResults.omuz_genisligi > 0 ? `${testResults.omuz_genisligi} cm` : '—'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-slate-600">Bel Genişliği</span>
                    <span className="font-semibold text-slate-900">
                      {testResults.bel_genisligi > 0 ? `${testResults.bel_genisligi} cm` : '—'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-slate-600">Omuz/Bel Oranı</span>
                    <span className="font-semibold text-slate-900">
                      {testResults.omuz_bel_orani > 0 ? testResults.omuz_bel_orani : '—'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-slate-600">Güvenilirlik</span>
                    <span className="font-semibold text-slate-900">
                      {testResults.confidence > 0 ? `${(testResults.confidence * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Vücut Tipi */}
            {testResults && (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">
                  Vücut Tipi Analizi
                </h3>
                
                <div className="text-center">
                  <div 
                    className="w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center text-white font-bold text-lg"
                    style={{ backgroundColor: getBodyTypeColor(testResults.vucut_tipi) }}
                  >
                    {testResults.vucut_tipi?.charAt(0)?.toUpperCase() || '?'}
                  </div>
                  
                  <h4 className="text-xl font-semibold text-slate-900 mb-2">
                    {testResults.vucut_tipi || 'Analiz Bekleniyor'}
                  </h4>
                  
                  <p className="text-slate-600 text-sm mb-4">
                    {getBodyTypeDescription(testResults.vucut_tipi)}
                  </p>
                  
                  {testResults.confidence > 0 && (
                    <div className="mt-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-slate-600">Güvenilirlik</span>
                        <span className="text-sm font-medium text-slate-900">
                          {(testResults.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${testResults.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Beslenme Özeti */}
            {testResults && testResults.diyet_onerileri && Object.keys(testResults.diyet_onerileri).length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                  <Utensils className="h-5 w-5 mr-2 text-green-600" />
                  Beslenme Özeti
                </h3>
                
                <div className="space-y-4">
                  {/* Vücut Tipi Özellikleri */}
                  {testResults.diyet_onerileri.ozellikler && (
                    <div>
                      <h4 className="font-semibold text-slate-800 mb-2">Özellikleriniz</h4>
                      <ul className="text-sm text-slate-600 space-y-1">
                        {testResults.diyet_onerileri.ozellikler.slice(0, 3).map((ozellik, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-green-500 mr-2">•</span>
                            {ozellik}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Beslenme İlkeleri */}
                  {testResults.diyet_onerileri.beslenme_ilkeleri && (
                    <div>
                      <h4 className="font-semibold text-slate-800 mb-2">Beslenme İlkeleri</h4>
                      <ul className="text-sm text-slate-600 space-y-1">
                        {testResults.diyet_onerileri.beslenme_ilkeleri.slice(0, 3).map((ilke, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            {ilke}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Kaçınılması Gerekenler */}
                  {testResults.diyet_onerileri.kacinilmasi_gerekenler && (
                    <div>
                      <h4 className="font-semibold text-slate-800 mb-2">Kaçınılması Gerekenler</h4>
                      <ul className="text-sm text-slate-600 space-y-1">
                        {testResults.diyet_onerileri.kacinilmasi_gerekenler.slice(0, 3).map((item, index) => (
                          <li key={index} className="flex items-start">
                            <span className="text-red-500 mr-2">•</span>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Sistem Durumu */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">
                Sistem Durumu
              </h3>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Test Durumu</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${testStatus.running ? 'bg-green-500' : testStatus.completed ? 'bg-blue-500' : 'bg-slate-300'}`} />
                    <span className="text-sm text-slate-900">
                      {testStatus.running ? 'Çalışıyor' : testStatus.completed ? 'Tamamlandı' : 'Bekliyor'}
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
                
                {testStatus.running && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-600">Kalan Süre</span>
                    <span className="text-sm text-slate-900">{testStatus.timeLeft}s</span>
                  </div>
                )}
                
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Kalori Hesaplama</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      calorieCalculationStatus.active ? 'bg-orange-500' : 
                      calorieCalculationStatus.completed ? 'bg-green-500' : 'bg-slate-300'
                    }`} />
                    <span className="text-sm text-slate-900">
                      {calorieCalculationStatus.active ? 'Aktif' : 
                       calorieCalculationStatus.completed ? 'Tamamlandı' : 'Bekliyor'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;