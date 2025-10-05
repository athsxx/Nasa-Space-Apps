#!/usr/bin/env python3
"""
Telegram Bot Command Handler
Handles incoming Telegram messages and bot commands for user connections
"""

import json
import logging
from telegram_service import TelegramAlertService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramBotHandler:
    """Handles Telegram bot commands and user interactions"""
    
    def __init__(self):
        """Initialize bot handler"""
        self.telegram_service = TelegramAlertService()
    
    def handle_webhook(self, update_data):
        """
        Handle incoming webhook update from Telegram
        
        Args:
            update_data: Telegram update JSON data
            
        Returns:
            Dict with processing result
        """
        try:
            if 'message' in update_data:
                message = update_data['message']
                chat_id = str(message['chat']['id'])
                
                if 'text' in message:
                    text = message['text'].strip()
                    
                    # Handle commands
                    if text.startswith('/'):
                        return self.handle_command(chat_id, text)
                    else:
                        # Handle regular text messages
                        return self.handle_text_message(chat_id, text)
                        
            return {'success': True, 'message': 'Update processed'}
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return {'success': False, 'message': f'Webhook error: {str(e)}'}
    
    def handle_command(self, chat_id, text):
        """
        Handle bot commands
        
        Args:
            chat_id: Telegram chat ID
            text: Command text
            
        Returns:
            Dict with command result
        """
        try:
            # Parse command and arguments
            parts = text.split(' ', 1)
            command = parts[0][1:]  # Remove the '/' prefix
            args = parts[1] if len(parts) > 1 else ""
            
            logger.info(f"Processing command: /{command} from chat {chat_id}")
            
            # Handle the command using telegram service
            result = self.telegram_service.handle_bot_command(chat_id, command, args)
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            return {'success': False, 'message': f'Command error: {str(e)}'}
    
    def handle_text_message(self, chat_id, text):
        """
        Handle regular text messages
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            
        Returns:
            Dict with message handling result
        """
        try:
            # Send a helpful response for regular messages
            help_msg = """
<b>🤖 NASA Air Quality Monitor Bot</b>

I understand commands better! Try these:

<b>Commands:</b>
/start - Begin setup or connect account
/help - Show help message
/status - Check your connection status
/unsubscribe - Stop receiving alerts

<b>To get started:</b>
1. Visit the NASA Air Quality web app
2. Click "Get updates on Telegram"
3. You'll be connected automatically!

Visit the app for detailed air quality data and predictions.
            """
            
            self.telegram_service.send_message(chat_id, help_msg)
            
            return {
                'success': True,
                'message': 'Help message sent for regular text'
            }
            
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            return {'success': False, 'message': f'Text handling error: {str(e)}'}

def main():
    """Test the bot handler"""
    handler = TelegramBotHandler()
    
    # Test with a sample update
    test_update = {
        "update_id": 123456,
        "message": {
            "message_id": 1,
            "date": 1234567890,
            "chat": {
                "id": 12345,
                "type": "private"
            },
            "text": "/start test_token_123"
        }
    }
    
    result = handler.handle_webhook(test_update)
    print(f"Test result: {result}")

if __name__ == "__main__":
    main()