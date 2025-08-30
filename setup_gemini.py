#!/usr/bin/env python3
"""
Simple Gemini API Key Setup Script for Aigie

This script helps you set up your Gemini API key for seamless Aigie usage.
"""

import os
import json
import getpass
from pathlib import Path

def setup_gemini_api_key():
    """Interactive setup for Gemini API key"""
    print("🚀 Aigie Gemini Setup")
    print("=" * 40)
    
    # Check if API key is already set
    existing_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if existing_key:
        print(f"✅ API key already found in environment: {existing_key[:8]}...")
        choice = input("Do you want to update it? (y/N): ").lower()
        if choice != 'y':
            print("Keeping existing API key.")
            return
    
    print("\n📋 Setup Options:")
    print("1. Set environment variable (recommended)")
    print("2. Create configuration file")
    print("3. Both")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice in ['1', '3']:
        setup_environment_variable()
    
    if choice in ['2', '3']:
        setup_config_file()
    
    print("\n✅ Setup complete!")
    print("\n💡 Usage:")
    print("   from aigie import AigieStateGraph")
    print("   graph = AigieStateGraph(enable_gemini_remediation=True)")
    print("   # Gemini will work automatically!")

def setup_environment_variable():
    """Set up environment variable"""
    print("\n🔑 Setting up environment variable...")
    
    # Get API key from user
    api_key = getpass.getpass("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Skipping environment variable setup.")
        return
    
    # Determine shell profile file
    home = Path.home()
    shell_profile = None
    
    if os.path.exists(home / ".zshrc"):
        shell_profile = home / ".zshrc"
        shell_name = "zsh"
    elif os.path.exists(home / ".bashrc"):
        shell_profile = home / ".bashrc"
        shell_name = "bash"
    elif os.path.exists(home / ".bash_profile"):
        shell_profile = home / ".bash_profile"
        shell_name = "bash"
    
    if shell_profile:
        # Add to shell profile
        export_line = f'\nexport GOOGLE_API_KEY="{api_key}"\n'
        
        with open(shell_profile, 'a') as f:
            f.write(export_line)
        
        print(f"✅ Added API key to {shell_profile}")
        print(f"   Restart your terminal or run: source {shell_profile}")
        
        # Set for current session
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ API key set for current session")
    else:
        print("⚠️  Could not find shell profile file")
        print("   Please manually add: export GOOGLE_API_KEY='your-key'")
        print("   to your shell configuration file")

def setup_config_file():
    """Set up configuration file"""
    print("\n📁 Setting up configuration file...")
    
    # Get API key from user
    api_key = getpass.getpass("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Skipping config file setup.")
        return
    
    # Create config directory
    config_dir = Path.home() / ".aigie"
    config_dir.mkdir(exist_ok=True)
    
    # Create config file
    config_file = config_dir / "config.json"
    config = {
        "api_key": api_key,
        "use_vertex_ai": False,
        "project_id": None,
        "location": "us-central1"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created configuration file: {config_file}")
    print("   This file will be automatically loaded by Aigie")

def test_setup():
    """Test if the setup is working"""
    print("\n🧪 Testing setup...")
    
    try:
        from aigie import AigieStateGraph
        
        # Try to create a graph with Gemini enabled
        graph = AigieStateGraph(enable_gemini_remediation=True)
        
        print("✅ Aigie imported successfully!")
        print("✅ Graph created with Gemini enabled!")
        
        # Check Gemini status
        if hasattr(graph, 'gemini_config'):
            config = graph.gemini_config
            if config["enabled"]:
                service_type = config["service_type"]
                if service_type == "developer_api":
                    print("✅ Gemini Developer API configured!")
                elif service_type == "vertex_ai":
                    print("✅ Gemini Vertex AI configured!")
                else:
                    print("⚠️  Gemini enabled but no configuration found")
            else:
                print("ℹ️  Gemini disabled")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Aigie: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing setup: {e}")
        return False

if __name__ == "__main__":
    try:
        setup_gemini_api_key()
        
        # Test the setup
        if test_setup():
            print("\n🎉 Setup successful! You can now use Aigie with Gemini AI.")
        else:
            print("\n⚠️  Setup completed but testing failed. Check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and try again.")
