/**
 * Audio visualizer functions
 */

// Function to create modern AI visualizer
function createModernVisualizer() {
    const container = document.createElement('div');
    container.classList.add('ai-visualizer-container');
    
    // Create SVG content
    container.innerHTML = `
    <svg class="ai-visualizer-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 120">
      <!-- Gradient definitions -->
      <defs>
        <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#4a86e8">
            <animate attributeName="stop-color" values="#4a86e8;#6b5ce7;#00bcd4;#4a86e8" dur="8s" repeatCount="indefinite" />
          </stop>
          <stop offset="100%" style="stop-color:#00bcd4">
            <animate attributeName="stop-color" values="#00bcd4;#4a86e8;#6b5ce7;#00bcd4" dur="8s" repeatCount="indefinite" />
          </stop>
        </linearGradient>
        
        <linearGradient id="circleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#4a86e8">
            <animate attributeName="stop-color" values="#4a86e8;#00bcd4;#6b5ce7;#4a86e8" dur="10s" repeatCount="indefinite" />
          </stop>
          <stop offset="100%" style="stop-color:#6b5ce7">
            <animate attributeName="stop-color" values="#6b5ce7;#4a86e8;#00bcd4;#6b5ce7" dur="10s" repeatCount="indefinite" />
          </stop>
        </linearGradient>
        
        <!-- Blur filter for glow effect -->
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>

      <!-- Central circle pulse -->
      <circle cx="150" cy="60" r="15" fill="url(#circleGradient)" filter="url(#glow)">
        <animate attributeName="r" values="15;20;15" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.8;1;0.8" dur="2s" repeatCount="indefinite" />
      </circle>

      <!-- Concentric rings -->
      <circle cx="150" cy="60" r="25" fill="none" stroke="url(#circleGradient)" stroke-width="1.5" opacity="0.6" filter="url(#glow)">
        <animate attributeName="r" values="25;35;25" dur="3s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.6;0.2;0.6" dur="3s" repeatCount="indefinite" />
      </circle>
      
      <circle cx="150" cy="60" r="40" fill="none" stroke="url(#circleGradient)" stroke-width="1" opacity="0.4" filter="url(#glow)">
        <animate attributeName="r" values="40;50;40" dur="4s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.4;0.1;0.4" dur="4s" repeatCount="indefinite" />
      </circle>
      
      <!-- Wave bars (left side) -->
      <g transform="translate(40, 60)">
        <!-- Wave bars -->
        <rect x="0" y="-30" width="6" height="60" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="20;60;20" dur="0.7s" repeatCount="indefinite" />
          <animate attributeName="y" values="-10;-30;-10" dur="0.7s" repeatCount="indefinite" />
        </rect>
        <rect x="12" y="-40" width="6" height="80" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="30;80;30" dur="0.9s" repeatCount="indefinite" />
          <animate attributeName="y" values="-15;-40;-15" dur="0.9s" repeatCount="indefinite" />
        </rect>
        <rect x="24" y="-35" width="6" height="70" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="35;70;35" dur="1.1s" repeatCount="indefinite" />
          <animate attributeName="y" values="-17.5;-35;-17.5" dur="1.1s" repeatCount="indefinite" />
        </rect>
        <rect x="36" y="-25" width="6" height="50" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="15;50;15" dur="0.8s" repeatCount="indefinite" />
          <animate attributeName="y" values="-7.5;-25;-7.5" dur="0.8s" repeatCount="indefinite" />
        </rect>
        <rect x="48" y="-20" width="6" height="40" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="10;40;10" dur="0.6s" repeatCount="indefinite" />
          <animate attributeName="y" values="-5;-20;-5" dur="0.6s" repeatCount="indefinite" />
        </rect>
      </g>
      
      <!-- Wave bars (right side) -->
      <g transform="translate(210, 60)">
        <!-- Mirror of left side -->
        <rect x="0" y="-20" width="6" height="40" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="10;40;10" dur="0.6s" repeatCount="indefinite" />
          <animate attributeName="y" values="-5;-20;-5" dur="0.6s" repeatCount="indefinite" />
        </rect>
        <rect x="12" y="-25" width="6" height="50" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="15;50;15" dur="0.8s" repeatCount="indefinite" />
          <animate attributeName="y" values="-7.5;-25;-7.5" dur="0.8s" repeatCount="indefinite" />
        </rect>
        <rect x="24" y="-35" width="6" height="70" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="35;70;35" dur="1.1s" repeatCount="indefinite" />
          <animate attributeName="y" values="-17.5;-35;-17.5" dur="1.1s" repeatCount="indefinite" />
        </rect>
        <rect x="36" y="-40" width="6" height="80" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="30;80;30" dur="0.9s" repeatCount="indefinite" />
          <animate attributeName="y" values="-15;-40;-15" dur="0.9s" repeatCount="indefinite" />
        </rect>
        <rect x="48" y="-30" width="6" height="60" rx="3" fill="url(#waveGradient)" filter="url(#glow)" opacity="0.9">
          <animate attributeName="height" values="20;60;20" dur="0.7s" repeatCount="indefinite" />
          <animate attributeName="y" values="-10;-30;-10" dur="0.7s" repeatCount="indefinite" />
        </rect>
      </g>
      
      <!-- Bottom wave path -->
      <path d="M60,85 Q90,70 120,85 Q150,100 180,85 Q210,70 240,85" 
            stroke="url(#waveGradient)" 
            stroke-width="3" 
            fill="none" 
            opacity="0.7"
            filter="url(#glow)">
        <animate attributeName="d" 
                values="M60,85 Q90,70 120,85 Q150,100 180,85 Q210,70 240,85;
                        M60,80 Q90,95 120,80 Q150,65 180,80 Q210,95 240,80;
                        M60,85 Q90,70 120,85 Q150,100 180,85 Q210,70 240,85" 
                dur="4s" 
                repeatCount="indefinite" />
      </path>
    </svg>`;
    
    return container;
}

// Function to show/hide visualizer
function toggleVisualizer(visualizer, isActive) {
    if (visualizer) {
        if (isActive) {
            visualizer.classList.add('active');
        } else {
            visualizer.classList.remove('active');
        }
    }
}

// Function to play Audio with visualizer
function playBase64AudioWithVisualizer(base64String, visualizer) {
    // Create an AudioContext
    const context = new (window.AudioContext || window.webkitAudioContext)();
    
    // Create analyzer for visualizations
    const analyzer = context.createAnalyser();
    analyzer.fftSize = 256;
    
    // Convert Base64 string to binary data
    let binaryString = atob(base64String);
    let len = binaryString.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Decode the audio data and play
    context.decodeAudioData(bytes.buffer, (audioBuffer) => {
        let source = context.createBufferSource();
        source.buffer = audioBuffer;
        
        // Connect to analyzer
        source.connect(analyzer);
        analyzer.connect(context.destination);
        
        // Start audio playback
        source.start(0);
        
        // Show visualizer
        toggleVisualizer(visualizer, true);
        
        // When audio ends, hide visualizer
        source.onended = function() {
            toggleVisualizer(visualizer, false);
        };
        
        // Fallback for older browsers that don't support onended
        setTimeout(() => {
            toggleVisualizer(visualizer, false);
        }, audioBuffer.duration * 1000 + 100); // Add small buffer
        
    }, (error) => {
        console.error("Error decoding audio:", error);
        toggleVisualizer(visualizer, false);
    });
}