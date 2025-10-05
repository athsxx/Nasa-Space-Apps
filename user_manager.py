#!/usr/bin/env python3
"""
User Management System for NASA Air Quality App
Handles user registration, Telegram bot connections, and user profiles
"""

import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserManager:
    """Manages user accounts and Telegram bot connections"""
    
    def __init__(self, users_file: str = "app_users.json"):
        """
        Initialize User Manager
        
        Args:
            users_file: JSON file to store user data
        """
        self.users_file = users_file
        self.users = self._load_users()
        
    def _load_users(self) -> Dict[str, Dict]:
        """Load users from JSON file"""
        try:
            import os
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
        return {}
    
    def _save_users(self) -> bool:
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
            return False
    
    def generate_user_token(self) -> str:
        """Generate a unique user token"""
        return str(uuid.uuid4())
    
    def create_user(self, ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """
        Create a new user account
        
        Args:
            ip_address: User's IP address (for analytics)
            user_agent: User's browser info (for analytics)
            
        Returns:
            Dict containing user info and connection token
        """
        try:
            user_token = self.generate_user_token()
            
            # Create anonymous fingerprint for privacy-friendly tracking
            fingerprint_data = f"{ip_address or 'unknown'}:{user_agent or 'unknown'}"
            user_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
            
            user_data = {
                'user_token': user_token,
                'fingerprint': user_fingerprint,
                'created_at': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'telegram_chat_id': None,
                'telegram_connected': False,
                'telegram_connected_at': None,
                'preferences': {
                    'location': None,
                    'alert_threshold': 101,
                    'notifications_enabled': True
                },
                'usage_stats': {
                    'total_queries': 0,
                    'last_query': None,
                    'favorite_locations': []
                }
            }
            
            self.users[user_token] = user_data
            
            if self._save_users():
                logger.info(f"New user created with token: {user_token[:8]}...")
                return {
                    'success': True,
                    'user_token': user_token,
                    'telegram_connection_url': f"http://t.me/Augustabot?start={user_token}",
                    'message': 'User account created successfully'
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save user account'
                }
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {
                'success': False,
                'message': f'Failed to create user: {str(e)}'
            }
    
    def get_user(self, user_token: str) -> Optional[Dict[str, Any]]:
        """Get user by token"""
        return self.users.get(user_token)
    
    def update_user_activity(self, user_token: str, activity_type: str = "query") -> bool:
        """Update user's last activity"""
        try:
            if user_token in self.users:
                self.users[user_token]['last_seen'] = datetime.now().isoformat()
                self.users[user_token]['usage_stats']['total_queries'] += 1
                self.users[user_token]['usage_stats']['last_query'] = datetime.now().isoformat()
                return self._save_users()
            return False
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
            return False
    
    def connect_telegram(self, user_token: str, telegram_chat_id: str) -> Dict[str, Any]:
        """
        Connect user's Telegram account
        
        Args:
            user_token: User's app token
            telegram_chat_id: Telegram chat ID from bot
            
        Returns:
            Dict with connection status
        """
        try:
            if user_token not in self.users:
                return {
                    'success': False,
                    'message': 'User not found'
                }
            
            # Update user with Telegram connection
            self.users[user_token]['telegram_chat_id'] = telegram_chat_id
            self.users[user_token]['telegram_connected'] = True
            self.users[user_token]['telegram_connected_at'] = datetime.now().isoformat()
            
            if self._save_users():
                logger.info(f"Telegram connected for user {user_token[:8]}... → chat_id: {telegram_chat_id}")
                return {
                    'success': True,
                    'message': 'Telegram successfully connected!',
                    'user_token': user_token,
                    'telegram_chat_id': telegram_chat_id
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save Telegram connection'
                }
                
        except Exception as e:
            logger.error(f"Error connecting Telegram: {e}")
            return {
                'success': False,
                'message': f'Telegram connection failed: {str(e)}'
            }
    
    def get_user_by_telegram(self, telegram_chat_id: str) -> Optional[Dict[str, Any]]:
        """Find user by Telegram chat ID"""
        for user_token, user_data in self.users.items():
            if user_data.get('telegram_chat_id') == telegram_chat_id:
                return {**user_data, 'user_token': user_token}
        return None
    
    def disconnect_telegram(self, user_token: str) -> Dict[str, Any]:
        """Disconnect user's Telegram account"""
        try:
            if user_token not in self.users:
                return {
                    'success': False,
                    'message': 'User not found'
                }
            
            self.users[user_token]['telegram_chat_id'] = None
            self.users[user_token]['telegram_connected'] = False
            self.users[user_token]['telegram_connected_at'] = None
            
            if self._save_users():
                return {
                    'success': True,
                    'message': 'Telegram disconnected successfully'
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save Telegram disconnection'
                }
                
        except Exception as e:
            logger.error(f"Error disconnecting Telegram: {e}")
            return {
                'success': False,
                'message': f'Telegram disconnection failed: {str(e)}'
            }
    
    def update_user_preferences(self, user_token: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        try:
            if user_token in self.users:
                self.users[user_token]['preferences'].update(preferences)
                return self._save_users()
            return False
        except Exception as e:
            logger.error(f"Error updating preferences: {e}")
            return False
    
    def get_connected_users(self) -> List[Dict[str, Any]]:
        """Get all users with Telegram connected"""
        connected_users = []
        for user_token, user_data in self.users.items():
            if user_data.get('telegram_connected'):
                connected_users.append({
                    **user_data,
                    'user_token': user_token
                })
        return connected_users
    
    def get_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        total_users = len(self.users)
        connected_users = len([u for u in self.users.values() if u.get('telegram_connected')])
        
        return {
            'total_users': total_users,
            'telegram_connected': connected_users,
            'connection_rate': f"{(connected_users/total_users*100):.1f}%" if total_users > 0 else "0%",
            'active_users_today': len([
                u for u in self.users.values() 
                if u.get('last_seen', '').startswith(datetime.now().strftime('%Y-%m-%d'))
            ])
        }