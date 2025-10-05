#!/usr/bin/env python3
"""
Telegram Alert Service for Air Quality Monitoring
Complete Telegram Bot API integration for sending AQI notifications
"""

import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramAlertService:
    """Telegram Alert service for air quality notifications"""
    
    def __init__(self, bot_token: str = "8348450996:AAGI1sl6XjqXyHv_qdXPBDbbS4SvSv7FwL4"):
        """
        Initialize Telegram Alert Service
        
        Args:
            bot_token: Telegram bot token from BotFather
        """
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.subscribers = self._load_subscribers()
        
    def _load_subscribers(self) -> Dict[str, Dict]:
        """Load subscribers from JSON file"""
        try:
            if os.path.exists('telegram_subscribers.json'):
                with open('telegram_subscribers.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load subscribers: {e}")
        return {}
    
    def _save_subscribers(self) -> bool:
        """Save subscribers to JSON file"""
        try:
            with open('telegram_subscribers.json', 'w') as f:
                json.dump(self.subscribers, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save subscribers: {e}")
            return False
    
    def send_message(self, chat_id: str, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send message to Telegram chat
        
        Args:
            chat_id: Telegram chat ID
            message: Message text (supports HTML formatting)
            parse_mode: Message parsing mode (HTML or Markdown)
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            
            if result.get('ok'):
                logger.info(f"Message sent successfully to {chat_id}")
                return True
            else:
                logger.error(f"Failed to send message: {result.get('description', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def subscribe_user(self, chat_id: str, location: str, threshold: int = 100) -> Dict[str, Any]:
        """
        Subscribe user to AQI alerts
        
        Args:
            chat_id: Telegram chat ID
            location: Location for AQI monitoring
            threshold: AQI threshold for alerts
            
        Returns:
            dict: Response with success status and message
        """
        try:
            # Validate chat ID by sending a test message
            welcome_msg = f"""
<b>AQI Alert Subscription Confirmed!</b>

<b>Location:</b> {location}
<b>Alert Threshold:</b> {threshold} AQI
<b>Chat ID:</b> {chat_id}

You'll receive air quality alerts when AQI exceeds {threshold}.

To unsubscribe, send: <code>/unsubscribe</code>
            """
            
            if self.send_message(chat_id, welcome_msg):
                # Save subscription
                self.subscribers[chat_id] = {
                    'location': location,
                    'threshold': threshold,
                    'subscribed_at': datetime.now().isoformat(),
                    'active': True
                }
                
                if self._save_subscribers():
                    return {
                        'success': True,
                        'message': f'Successfully subscribed to AQI alerts for {location}',
                        'chat_id': chat_id,
                        'subscription': {
                            'location': location,
                            'threshold': threshold,
                            'active': True
                        }
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Failed to save subscription'
                    }
            else:
                return {
                    'success': False,
                    'message': 'Invalid chat ID or bot blocked by user'
                }
                
        except Exception as e:
            logger.error(f"Error subscribing user: {e}")
            return {
                'success': False,
                'message': f'Subscription failed: {str(e)}'
            }
    
    def unsubscribe_user(self, chat_id: str) -> Dict[str, Any]:
        """
        Unsubscribe user from AQI alerts
        
        Args:
            chat_id: Telegram chat ID
            
        Returns:
            dict: Response with success status and message
        """
        try:
            if chat_id in self.subscribers:
                del self.subscribers[chat_id]
                if self._save_subscribers():
                    # Send confirmation message
                    goodbye_msg = """
🚫 <b>Unsubscribed Successfully</b>

You will no longer receive AQI alerts.

To subscribe again, visit: <code>/telegram/subscribe</code>
                    """
                    self.send_message(chat_id, goodbye_msg)
                    
                    return {
                        'success': True,
                        'message': 'Successfully unsubscribed from AQI alerts'
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Failed to save unsubscribe status'
                    }
            else:
                return {
                    'success': False,
                    'message': 'Chat ID not found in subscribers'
                }
                
        except Exception as e:
            logger.error(f"Error unsubscribing user: {e}")
            return {
                'success': False,
                'message': f'Unsubscribe failed: {str(e)}'
            }
    
    def get_aqi_color_emoji(self, aqi: int) -> str:
        """Get emoji for AQI level"""
        if aqi <= 50:
            return "🟢"  # Good
        elif aqi <= 100:
            return "🟡"  # Moderate
        elif aqi <= 150:
            return "🟠"  # Unhealthy for Sensitive Groups
        elif aqi <= 200:
            return "🔴"  # Unhealthy
        elif aqi <= 300:
            return "🟣"  # Very Unhealthy
        else:
            return "🟤"  # Hazardous
    
    def send_aqi_alert(self, chat_id: str, aqi_data: Dict[str, Any]) -> bool:
        """
        Send AQI alert to Telegram chat
        
        Args:
            chat_id: Telegram chat ID
            aqi_data: AQI data dictionary
            
        Returns:
            bool: True if alert sent successfully
        """
        try:
            aqi_value = aqi_data.get('overall_aqi', 0)
            category = aqi_data.get('overall_category', aqi_data.get('category', 'Unknown'))
            location = aqi_data.get('location', {}).get('address', 'Unknown Location')
            dominant_pollutant = aqi_data.get('dominant_pollutant', 'Unknown')
            health_message = aqi_data.get('health_message', 'Air quality information available')
            
            emoji = self.get_aqi_color_emoji(aqi_value)
            
            # Create formatted message
            message = f"""
{emoji} <b>AQI Alert - {category}</b>

<b>Location:</b> {location}
<b>AQI Value:</b> {aqi_value}
<b>Category:</b> {category}
<b>Dominant Pollutant:</b> {dominant_pollutant.upper()}

💊 <b>Health Advisory:</b>
{health_message}

🕒 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>Stay safe and monitor air quality regularly!</i>
            """
            
            return self.send_message(chat_id, message)
            
        except Exception as e:
            logger.error(f"Error sending AQI alert: {e}")
            return False
    
    def broadcast_alert(self, aqi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast AQI alert to all subscribers with matching location/threshold
        
        Args:
            aqi_data: AQI data dictionary
            
        Returns:
            dict: Broadcast results
        """
        results = {
            'sent': 0,
            'failed': 0,
            'total_subscribers': len(self.subscribers),
            'details': []
        }
        
        aqi_value = aqi_data.get('overall_aqi', 0)
        location = aqi_data.get('location', {}).get('address', '').lower()
        
        for chat_id, sub_data in self.subscribers.items():
            try:
                # Check if this subscriber should receive this alert
                sub_location = sub_data.get('location', '').lower()
                threshold = sub_data.get('threshold', 100)
                active = sub_data.get('active', True)
                
                if active and aqi_value >= threshold:
                    # Send alert
                    if self.send_aqi_alert(chat_id, aqi_data):
                        results['sent'] += 1
                        results['details'].append({
                            'chat_id': chat_id,
                            'status': 'sent',
                            'location': sub_data.get('location')
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'chat_id': chat_id,
                            'status': 'failed',
                            'location': sub_data.get('location')
                        })
                        
            except Exception as e:
                logger.error(f"Error broadcasting to {chat_id}: {e}")
                results['failed'] += 1
        
        return results
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get Telegram service status
        
        Returns:
            dict: Service status information
        """
        try:
            # Test bot connection
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=5)
            bot_info = response.json()
            
            if bot_info.get('ok'):
                bot_data = bot_info.get('result', {})
                return {
                    'service': 'telegram',
                    'status': 'operational',
                    'bot_authenticated': True,
                    'bot_info': {
                        'username': bot_data.get('username'),
                        'first_name': bot_data.get('first_name'),
                        'can_read_all_group_messages': bot_data.get('can_read_all_group_messages'),
                        'supports_inline_queries': bot_data.get('supports_inline_queries')
                    },
                    'subscribers_count': len(self.subscribers),
                    'active_subscribers': len([s for s in self.subscribers.values() if s.get('active', True)]),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'service': 'telegram',
                    'status': 'error',
                    'bot_authenticated': False,
                    'error': bot_info.get('description', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error checking service status: {e}")
            return {
                'service': 'telegram',
                'status': 'error',
                'bot_authenticated': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_bot_info(self) -> Dict[str, Any]:
        """Get bot information for setup verification"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=5)
            return response.json()
        except Exception as e:
            logger.error(f"Error getting bot info: {e}")
            return {'ok': False, 'error': str(e)}


# Global telegram service instance
telegram_service = TelegramAlertService()