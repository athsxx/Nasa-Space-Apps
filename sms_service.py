#!/usr/bin/env python3
"""
SMS Alert Service using Twilio
Provides air quality alerts via SMS notifications
"""

import json
import datetime
from twilio.rest import Client
from typing import Dict, List, Optional

# Twilio Configuration
TWILIO_SID = "ACa65323829a21cf3daa9e766316e0eac9"
TWILIO_AUTH_TOKEN = "4bb2fae83628ffc24c55805530aa19f1"
# For Twilio trial accounts, you need to get your trial number from the console
# Common trial numbers start with +15005550006 (test number) or similar
TWILIO_PHONE_NUMBER = "+15005550006"  # Twilio test number - replace with your actual number

class SMSAlertService:
    """SMS Alert service for air quality notifications"""
    
    def __init__(self):
        """Initialize Twilio client and get phone number"""
        try:
            self.client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            self.subscribers = {}  # In production, use a database
            
            # Try to get the first available phone number from the account
            self.twilio_number = self._get_twilio_phone_number()
            
            if self.twilio_number:
                print(f"✅ SMS Alert Service initialized with number: {self.twilio_number}")
            else:
                print("⚠️ SMS Alert Service initialized but no phone number found")
                
        except Exception as e:
            print(f"❌ SMS Service initialization failed: {e}")
            self.client = None
            self.twilio_number = None
    
    def _get_twilio_phone_number(self):
        """Get the first available Twilio phone number from the account"""
        try:
            if not self.client:
                return None
                
            # Get incoming phone numbers from the account
            phone_numbers = self.client.incoming_phone_numbers.list(limit=1)
            
            if phone_numbers:
                return phone_numbers[0].phone_number
            else:
                # For trial accounts without phone numbers, we can't send SMS
                print("⚠️ No phone numbers found in Twilio account")
                print("💡 To enable SMS functionality:")
                print("   1. Go to https://console.twilio.com/")
                print("   2. Get a phone number from Phone Numbers > Manage > Buy a number")
                print("   3. For trial accounts, you can get a free number")
                return None
                
        except Exception as e:
            print(f"❌ Could not retrieve phone number: {e}")
            return None
    
    def subscribe_user(self, phone_number: str, location: str = "", alert_threshold: int = 100) -> Dict:
        """
        Subscribe a user to SMS alerts
        
        Args:
            phone_number: User's phone number (format: +1234567890)
            location: User's preferred location for alerts
            alert_threshold: AQI threshold for alerts (default: 100 - Unhealthy for Sensitive Groups)
        
        Returns:
            Dict with subscription status
        """
        try:
            # Validate phone number format
            if not phone_number.startswith('+'):
                phone_number = '+1' + phone_number.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
            
            # Store subscription (in production, save to database)
            self.subscribers[phone_number] = {
                'location': location,
                'alert_threshold': alert_threshold,
                'subscribed_at': datetime.datetime.now().isoformat(),
                'active': True
            }
            
            # Send welcome SMS
            welcome_message = f"""
🌍 NASA Air Quality Alerts

Welcome! You're now subscribed to air quality alerts.
📍 Location: {location or 'Auto-detect'}
🚨 Alert threshold: {alert_threshold} AQI
📱 Reply STOP to unsubscribe anytime

Stay safe! 🌟
"""
            
            self.send_sms(phone_number, welcome_message.strip())
            
            return {
                "success": True,
                "message": "Successfully subscribed to SMS alerts",
                "phone": phone_number,
                "threshold": alert_threshold
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Subscription failed: {str(e)}"
            }
    
    def unsubscribe_user(self, phone_number: str) -> Dict:
        """Unsubscribe a user from SMS alerts"""
        try:
            if phone_number in self.subscribers:
                self.subscribers[phone_number]['active'] = False
                
                goodbye_message = """
🌍 NASA Air Quality Alerts

You've been unsubscribed from air quality alerts.
Thank you for using our service! 

Reply START to resubscribe anytime.
"""
                
                self.send_sms(phone_number, goodbye_message.strip())
                
                return {
                    "success": True,
                    "message": "Successfully unsubscribed"
                }
            else:
                return {
                    "success": False,
                    "error": "Phone number not found in subscribers"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Unsubscribe failed: {str(e)}"
            }
    
    def send_aqi_alert(self, phone_number: str, aqi_data: Dict) -> bool:
        """
        Send AQI alert SMS to a subscriber
        
        Args:
            phone_number: Subscriber's phone number
            aqi_data: AQI data dictionary
        
        Returns:
            bool: Success status
        """
        try:
            overall_aqi = aqi_data.get('overall_aqi', 0)
            location = aqi_data.get('location', {}).get('address', 'Unknown Location')
            category = aqi_data.get('overall_category', 'Unknown')
            dominant_pollutant = aqi_data.get('dominant_pollutant', 'N/A').upper()
            
            # Get alert emoji based on AQI level
            alert_emoji = self._get_aqi_emoji(overall_aqi)
            
            message = f"""
{alert_emoji} AIR QUALITY ALERT {alert_emoji}

📍 {location}
🚨 AQI: {overall_aqi} ({category})
🔬 Main pollutant: {dominant_pollutant}

{self._get_health_advice(overall_aqi)}

Time: {datetime.datetime.now().strftime('%I:%M %p')}
"""
            
            return self.send_sms(phone_number, message.strip())
            
        except Exception as e:
            print(f"❌ Failed to send AQI alert: {e}")
            return False
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """
        Send SMS message via Twilio
        
        Args:
            phone_number: Recipient's phone number
            message: Message content
            
        Returns:
            bool: Success status
        """
        if not self.client:
            print("❌ Twilio client not initialized")
            return False
            
        if not self.twilio_number:
            print("❌ No Twilio phone number available")
            return False
        
        try:
            message_obj = self.client.messages.create(
                body=message,
                from_=self.twilio_number,
                to=phone_number
            )
            
            print(f"✅ SMS sent to {phone_number}: {message_obj.sid}")
            return True
            
        except Exception as e:
            error_str = str(e)
            print(f"❌ SMS failed to {phone_number}: {e}")
            print(f"   From: {self.twilio_number}")
            print(f"   To: {phone_number}")
            
            # Provide helpful error messages for common issues
            if "not a verified number" in error_str or "not authorized" in error_str:
                print("💡 Trial Account Limitation:")
                print("   - For trial accounts, you can only send SMS to verified numbers")
                print("   - Go to https://console.twilio.com/ > Phone Numbers > Verified Caller IDs")
                print("   - Add and verify the recipient's phone number")
            elif "Unable to create record" in error_str and "Premium rate" in error_str:
                print("💡 Phone Number Restriction:")
                print("   - Cannot send to premium rate or special service numbers")
                print("   - Use a regular mobile phone number instead")
            elif "not a valid phone number" in error_str:
                print("💡 Phone Number Format:")
                print("   - Use format: +1234567890 (with country code)")
                print("   - Ensure the number is valid and properly formatted")
            
            return False
    
    def check_and_send_alerts(self, aqi_data: Dict) -> List[str]:
        """
        Check all subscribers and send alerts if needed
        
        Args:
            aqi_data: Current AQI data
            
        Returns:
            List of phone numbers that received alerts
        """
        alerts_sent = []
        overall_aqi = aqi_data.get('overall_aqi', 0)
        
        for phone_number, subscription in self.subscribers.items():
            if not subscription.get('active', False):
                continue
                
            threshold = subscription.get('alert_threshold', 100)
            
            if overall_aqi >= threshold:
                if self.send_aqi_alert(phone_number, aqi_data):
                    alerts_sent.append(phone_number)
        
        return alerts_sent
    
    def _get_aqi_emoji(self, aqi: int) -> str:
        """Get emoji based on AQI level"""
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
            return "🚨"  # Hazardous
    
    def _get_health_advice(self, aqi: int) -> str:
        """Get health advice based on AQI level"""
        if aqi <= 50:
            return "✅ Air quality is good. Enjoy outdoor activities!"
        elif aqi <= 100:
            return "⚠️ Moderate air quality. Sensitive people should limit outdoor activities."
        elif aqi <= 150:
            return "🚫 Unhealthy for sensitive groups. Limit outdoor activities."
        elif aqi <= 200:
            return "⛔ Unhealthy air quality. Avoid outdoor activities."
        elif aqi <= 300:
            return "🚨 Very unhealthy! Stay indoors and avoid outdoor activities."
        else:
            return "🆘 HAZARDOUS! Stay indoors. Seek medical attention if experiencing symptoms."
    
    def get_subscriber_count(self) -> int:
        """Get total number of active subscribers"""
        return sum(1 for sub in self.subscribers.values() if sub.get('active', False))
    
    def get_subscriber_info(self, phone_number: str) -> Optional[Dict]:
        """Get subscriber information"""
        return self.subscribers.get(phone_number)
    
    def get_service_status(self) -> Dict:
        """Get SMS service status"""
        active_subscribers = sum(1 for sub in self.subscribers.values() if sub.get('active', True))
        
        return {
            "service_active": self.client is not None,
            "twilio_number": self.twilio_number,
            "total_subscribers": len(self.subscribers),
            "active_subscribers": active_subscribers,
            "messages_sent_today": 0  # Would implement with proper logging
        }
    
    def test_twilio_connection(self) -> Dict:
        """Test Twilio connection and account details"""
        try:
            if not self.client:
                return {"success": False, "error": "Twilio client not initialized"}
            
            # Get account info
            account = self.client.api.accounts(TWILIO_SID).fetch()
            
            # Get phone numbers
            phone_numbers = self.client.incoming_phone_numbers.list(limit=5)
            numbers = [num.phone_number for num in phone_numbers]
            
            return {
                "success": True,
                "account_sid": account.sid,
                "account_status": account.status,
                "phone_numbers": numbers,
                "current_number": self.twilio_number
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global SMS service instance
sms_service = SMSAlertService()