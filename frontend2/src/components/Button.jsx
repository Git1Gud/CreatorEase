import React from 'react';

const Button = ({ children, onClick, className, ...props }) => {
  return (
    <button 
      className={`custom-button ${className || ''}`} 
      onClick={onClick}
      {...props} // Spreads other valid button attributes like 'disabled', 'type', etc.
    >
      {children}
    </button>
  );
}

export default Button; 