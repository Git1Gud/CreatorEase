import React, { createContext, useContext, useEffect, useState } from "react";
import { io } from "socket.io-client";
import { mergeTrimItems } from "../utils";

import { useVideoEditor } from "./VideoEditorContext";

const SocketContext = createContext(undefined);

let socket;

export function SocketContextProvider({ children }) {
  const [userId, setUserId] = useState(null);
  const [socketId, setSocketId] = useState(null);
  const [roomInfo, setRoomInfo] = useState(null);
 const { // Destructure directly, types come from useVideoEditor's return type
       
       trimHistory,
       setTrimHistory,
       
       
     } = useVideoEditor();

  useEffect(() => {
    // connect to backend
    socket = io(import.meta.env.VITE_BACKEND_URL || "http://localhost:8000");

    socket.on("connect", () => {
      console.log("Connected to server:", socket.id);
    });

    socket.on("connected", ({ userId, socketId }) => {
      console.log("Setup complete", userId, socketId);
      setUserId(userId);
      setSocketId(socketId);
    });

    socket.on("room joined", ({ roomId, leader }) => {
      console.log(`Joined room ${roomId}, leader is ${leader}`);
      setRoomInfo({ roomId, leader });
    });


    socket.on("validate",({newTrim}) =>{
        console.log("checking for trim to be non overlapping");
        console.log(trimHistory);

        const overlap=trimHistory.some((trim)=>{
            (newTrim.start>=trim.start && newTrim.start<=newTrim.end) || ( newTrim.end>=trim.start && newTrim.end>=trim.end )
                
            });


           
        if(!overlap){
            console.log("trim was indeed not overlapping");
        socket.emit("update",({newTrim,userId})); 
        }
        else {
            console.log("trim was  overlapping");
        
        }
         
    })


    socket.on("update",({newTrim}) =>{
        setTrimHistory(prev => mergeTrimItems(prev, [newTrim])); 
    })

    socket.on("error", (err) => {
      console.error("Server error:", err);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const setup = (maybeUserId) => {
    socket.emit("setup", { userId: maybeUserId });
  };

  const joinRoom = (roomId = null) => {
    if (!userId) {
      console.warn("Must call setup before joinRoom");
      return;
    }
    socket.emit("join room", { userId, roomId });
  };

  const updateWithValidation=(newTrim)=>{
    if (!roomInfo.roomId) {
      console.warn("Must call join room before validation");
      return;
    }
    socket.emit("validate", {newTrim,leader:roomInfo.leader,room:roomInfo.room} );
  }

  const value = {
    userId,
    socketId,
    roomInfo,
    setup,
    joinRoom,
    updateWithValidation
  };

  return (
    <SocketContext.Provider value={value}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket() {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error("socket must be used within a sockeprovider");
  }
  return context;
}
