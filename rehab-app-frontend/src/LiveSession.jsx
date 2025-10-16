/* === LiveSssion.jsx === */
import { useRef, useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Pose } from "@mediapipe/pose";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { POSE_CONNECTIONS } from "@mediapipe/pose";
import { Pie,Bar } from "react-chartjs-2";

import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,                                       
  LinearScale,                                       
  BarElement                                          
} from "chart.js";
import "./App.css";

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

/* ──────────────────────────────────────────────
   constants
   ──────────────────────────────────────────── */
const WS_URL = "ws://localhost:8000/ws/infer";

const EXERCISE_MAP = {
  1: "Arm abduction",
  2: "Arm VW",
  3: "Push-ups",
  4: "Leg abduction",
  5: "Leg lunge",
  6: "Squats",
};

const JOINT_TRIPLETS = {
  "left elbow"   : [11, 13, 15],
  "right elbow"  : [12, 14, 16],
  "left shoulder": [11, 13, 23],
  "right shoulder":[12, 14, 24],
  "left hip"     : [11, 23, 25],
  "right hip"    : [12, 24, 26],
  "left knee"    : [23, 25, 27],
  "right knee"   : [24, 26, 28],
  "spine"        : [11, 12, 23],
  "head"         : [0, 11, 12],
  "left wrist"   : [13, 15, 19],
  "right wrist"  : [14, 16, 20],
  "left ankle"   : [25, 27, 31],
  "right ankle"  : [26, 28, 32],
};

const SEQ_LEN = 16;

/* ──────────────────────────────────────────────
   component
   ──────────────────────────────────────────── */
export default function App() {
  /* refs that survive re-renders */
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const poseRef   = useRef(null);
  const wsRef     = useRef(null);
  const bufRef    = useRef([]);
  const synthRef = useRef(window.speechSynthesis);
  const hlRef = useRef(new Set());
  const lastSpokenRef = useRef("");     
  const speakingRef = useRef(false);  
  const focusMsgRef     = useRef("");  
  

  /* UI state */
  const [cameraReady,      setCameraReady ] = useState(false);
  const [selectedExercise, setExercise    ] = useState(1);
  const [running,          setRunning     ] = useState(false);
  const [feedback,         setFeedback    ] = useState("Waiting…");
  const [suggestion,       setSuggestion  ] = useState("");
  const [kpi,              setKpi         ] = useState({ avg:"0.0", correct:0, total:0 });
  const [sessionSummary,   setSummary     ] = useState(null);
  const [fbType, setFbType] = useState("init");
  const [jointMean,        setJointMean   ] = useState(Array(14).fill(0));  
  const [focusMsg,         setFocusMsg    ] = useState(""); 
  const [chartKey,         setChartKey    ] = useState(0);

   /* navigate/location hooks */
  const navigate   = useNavigate();         
  const location   = useLocation();        

  /* if we returned from /summary with a pre-chosen exercise */
 useEffect(() => {
  if (location.state?.selectedExercise) {
    setExercise(location.state.selectedExercise);
    }
  }, [location.state]);

  // Speak whenever feedback or suggestion changes
  useEffect(() => {
    if (!running) return;                         // nothing if session stopped
  
    // ───────── priority logic ─────────
    //  1. Speak suggestion if it exists (non-empty string)
    //  2. Otherwise speak feedback
    const textToSpeak = suggestion.trim() || feedback.trim();
    if (!textToSpeak) return;                     // both empty → do nothing
     
    lastSpokenRef.current = textToSpeak;                        

    const speak = () => {
      const utter = new SpeechSynthesisUtterance(textToSpeak);
      speakingRef.current = true;                              
      utter.onend = () => { speakingRef.current = false; };     
      synthRef.current.speak(utter);
    };

    if (speakingRef.current) {
      /* queue once the current utterance ends */
      const handler = () => {
        synthRef.current.removeEventListener("end", handler);
        speak();
      };
      synthRef.current.addEventListener("end", handler);        // ★ CHANGED
    } else {
      speak();
    }
  }, [suggestion, feedback, running]); // keep both in dep list

  /* ────────────────────────── helpers ─────────────────────────── */
  /**
   * Clean-up helper.
   * @param {boolean} closeWs – also close the WebSocket? (default true)
   */
  const cleanUp = (closeWs = true) => {
    cameraRef.current?.stop();
    poseRef.current?.reset?.();
    if (closeWs) wsRef.current?.close?.();          // close only if asked
    if (videoRef.current) videoRef.current.srcObject = null;
    synthRef.current.cancel();

    cameraRef.current = null;
    poseRef.current   = null;
    if (closeWs) wsRef.current = null;

    bufRef.current    = [];
    setCameraReady(false);
  };

  /** initialise Pose (only after WS is open) */
  const initPoseAndCamera = () => {
    const pose = new Pose({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
    });
    pose.setOptions({
      modelComplexity: 2,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults(res => {
      if (!videoRef.current || !canvasRef.current) return;

      if (!cameraReady) setCameraReady(true);

      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      if (res.poseLandmarks) {
        drawConnectors(ctx, res.poseLandmarks, POSE_CONNECTIONS,
                       { color: "#00FF00", lineWidth: 4 });
        // draw landmarks – highlight the specified ones in amber
        const highlightOnly = hlRef.current.size > 0;
        res.poseLandmarks.forEach((lm, i) => {
          const isHL = hlRef.current.has(i);
          if (!highlightOnly || isHL) {
            drawLandmarks(ctx, [lm], {
              color : isHL ? "#FFA500" : "#87CEEB",   // amber vs red
              radius: isHL ? 6 : 4,
            });
          }
        });

        const w = res.poseWorldLandmarks;
        if (w?.length === 33) {
          bufRef.current.push(w.map(lm => [lm.x, lm.y, lm.z]));
          if (bufRef.current.length >= SEQ_LEN && wsRef.current?.readyState === 1) {
            wsRef.current.send(JSON.stringify({
              label: "keypoint_sequence",
              exercise_id: selectedExercise,
              keypoints: bufRef.current.slice(0, SEQ_LEN),
            }));
            bufRef.current = [];
          }
        }
      }
    });

    poseRef.current = pose;

    /* camera */
    cameraRef.current = new Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current.readyState >= 2) {
          await pose.send({ image: videoRef.current });
        }
      },
      width : 640,
      height: 480,
    });
    cameraRef.current.start();
  };

  /* ────────────────────────── handlers ─────────────────────────── */
  const startExercise = () => {
    /* clean any previous run */
    cleanUp();

    /* reset UI */
    setRunning(true);
    setChartKey(k => k + 1);
    setFeedback("Initialising…");
    setFbType("init"); 
    setSuggestion("");
    setKpi({ avg:"0.0", correct:0, total:0 });
    setSummary(null);
    setJointMean(Array(14).fill(0));              
    setFocusMsg("");                             

    /* create a brand-new WebSocket */
    wsRef.current = new WebSocket(WS_URL);

    wsRef.current.onopen = () => {
      console.log("WebSocket connected");
      initPoseAndCamera();          // ← only now start video processing
    };

    wsRef.current.onclose = () => {
      console.log("WebSocket disconnected");
    };

    wsRef.current.onmessage = ev => {
      const d = JSON.parse(ev.data);
      if (d.type === "progress") {
        setFeedback(d.feedback);
        // classify feedback
        if (/right track/i.test(d.feedback))       setFbType("good");
        else if (/initialising/i.test(d.feedback)) setFbType("init");
        else                                       setFbType("bad");
        setSuggestion(d.suggestion);
        setKpi({
          avg    : d.avg_error.toFixed(1),
          correct: d.correct,
          total  : d.total,
        });
        /* update highlight indices based on “Adjust your …” suggestion */
        if (typeof d.suggestion === "string") {
          const m = d.suggestion.match(/adjust your (.+?) properly/i);
          if (m) {
            const joints = m[1]
              .toLowerCase()
              .split(/,| and /)
              .map(s => s.trim())
              .filter(Boolean);

            const inds = new Set();
            joints.forEach(j => {
              (JOINT_TRIPLETS[j] || []).forEach(i => inds.add(i));
            });
            hlRef.current = inds;        // save for drawing loop
          } else {
            hlRef.current = new Set();   // no highlight
          }
        }
        /* ───  histogram data ─── */
        if (d.joint_errors_mean) setJointMean([...d.joint_errors_mean]);   
        if (d.top_joints) {
          const msg = `You should focus on correcting your ${d.top_joints.join(", ")} more`;
          setFocusMsg(msg);          // for live UI (if ever re-enable it there)
          focusMsgRef.current = msg; // ← keep the latest value for summary page 
        } 
      } else if (d.type === "summary") {
        /* summary from backend → go to dashboard */
        const kpiPack = {
          avg    : (d.joint_errors?.length
                     ? (d.joint_errors.reduce((a,b)=>a+b,0) /
                        d.joint_errors.length).toFixed(1)
                     : "0.0"),
          correct: d.correct,
          total  : d.total,
        };

        navigate("/summary", {
          state:{
            kpi        : kpiPack,
            jointMean  : d.joint_errors ?? Array(14).fill(0),
            selectedExercise,
            focusMsg : focusMsgRef.current,
          },
        });

        /* now it's safe to close the socket and clear everything */
        wsRef.current?.close?.();
        wsRef.current = null;
        cleanUp();                  // full clean-up
      }
    };
  };

  const stopExercise = () => {
    /* tell backend to finish */
    if (wsRef.current?.readyState === 1) {
      wsRef.current.send(JSON.stringify({ label: "stop" }));
    }

    cleanUp(false);                 // keep WebSocket open for summary

    setRunning(false);
    setFeedback("Stopped – preparing summary…");
    setSuggestion("");
  };

  /* ────────────────────────── chart data ─────────────────────────── */
  const pieData = {
    labels: ["Correct", "Incorrect"],
    datasets: [{
      data: [kpi.correct, Math.max(kpi.total - kpi.correct, 0)],
      backgroundColor     : ["#4caf50", "#f44336"],
      hoverBackgroundColor: ["#66bb6a", "#e57373"],
      borderWidth: 1,
    }],
  };

  const barData = {                                       
    labels: [
      "L-elbow","R-elbow","L-shoulder","R-shoulder","L-hip","R-hip",
      "L-knee","R-knee","Spine","Head","L-wrist","R-wrist","L-ankle","R-ankle"
    ],
    datasets: [{
      data: jointMean,
      backgroundColor: "#2196f3",
      borderWidth: 1,
    }],
  };

  /* ────────────────────────── UI ─────────────────────────── */
  return (
    <div style={{ textAlign:"center", padding:20 }}>
      <h1>RehabApp: AI Physiotherapist</h1>

      {/* selector + start/stop */}
      <div style={{ marginBottom:20 }}>
        <label>
          Select Exercise:&nbsp;
          <select
            className="exercise-select"
            value={selectedExercise}
            onChange={e => setExercise(+e.target.value)}
            disabled={running}
          >
            {Object.entries(EXERCISE_MAP).map(([id,name]) =>
              <option key={id} value={id}>{name}</option>
            )}
          </select>
        </label>

        {running ? (
          <button className="btn btn-stop"  style={{ marginLeft:20 }} onClick={stopExercise}>Stop</button>
        ) : (
          <button className="btn btn-start" style={{ marginLeft:20 }} onClick={startExercise}>Start</button>
        )}
      </div>

      {/* video + KPI */}
      <div className="content-wrapper">
        {/* video */}
        <div className="video-container">
          <video
            ref={videoRef}
            width="640"
            height="480"
            style={{ position:"absolute", inset:0, objectFit:"cover",
                     width:"100%", height:"100%",
                     visibility: cameraReady ? "visible" : "hidden" }}
            autoPlay muted playsInline
          />
          <canvas
            ref={canvasRef}
            width="640"
            height="480"
            style={{ position:"absolute", inset:0, width:"100%", height:"100%",
                     visibility: cameraReady ? "visible" : "hidden" }}
          />

          {!cameraReady && (
            <div className="video-placeholder">
              <p>{running
                  ? "Video is initialising, please wait…"
                  : <>Click <strong>Start</strong> to begin your exercise</>}
              </p>
            </div>
          )}
        </div>

        {/* KPI + chart */}
        <div className="kpi-container">
          {running || sessionSummary ? (
            <>
              <div className="kpi-grid">
                <div className="kpi-card">
                  <h4>Avg error</h4>
                  <span>{kpi.avg}°</span>
                </div>
                <div className="kpi-card">
                  <h4>% correct</h4>
                  <span>{kpi.total ? Math.round(100*kpi.correct/kpi.total) : 0}%</span>
                </div>
              </div>

              {kpi.total > 0 && (
                <div style={{ width:220, margin:"20px auto" }}>
                  <Pie key={`pie-${chartKey}`}
                    data={pieData}
                    options={{
                      plugins:{ legend:{ display:true, position:"bottom" } },
                      maintainAspectRatio:false,
                    }}
                  />
                </div>
              )}
              {/* ─── NEW histogram ─── */}
              <div style={{ width:420, height:260, margin:"20px auto" }}>  
                <Bar key={`bar-${chartKey}`}                                                  
                  data={barData}                                      
                  options={{                                         
                    plugins:{ legend:{ display:false } },            
                    scales : { y:{ beginAtZero:true,
                                    title:{ text:"° deviation", display:true } } },
                    maintainAspectRatio:false,
                  }}
                />                                                   
              </div>                                               
              {/* {!!focusMsg && <p style={{fontWeight:600}}>{focusMsg}</p>}  */}
            </>
          ) : (
            <div className="chart-placeholder"><p>Your KPIs will appears here</p></div>
          )}
        </div>
      </div>

      <h2 className={`feedback-${fbType}`} style={{ marginTop:20 }}>
      {feedback}
      </h2>
      {suggestion && (<h3 className="suggestion-text">Suggestion: {suggestion}</h3>)}
      {/* ───────── NEW ───────── */}
      <button
        className="btn btn-feedback"   // ← changed
        style={{ marginTop: 30 }}
        onClick={() => window.location.href = "/feedback"}
      >
        Your Feedbacks
      </button>
    </div>
  );
}
