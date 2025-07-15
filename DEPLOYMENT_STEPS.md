# DomainAI Deployment - Step by Step Guide

This guide provides exact commands and steps to deploy your DomainAI application and make it publicly accessible.

## Option 1: Deploying to an AWS EC2 Instance (Recommended)

### Step 1: Launch an AWS EC2 Instance

1. Log in to your AWS Management Console
2. Navigate to EC2 Dashboard
3. Click "Launch Instance"
4. Choose an Amazon Machine Image (AMI):
   - Select "Ubuntu Server 22.04 LTS"
5. Choose an Instance Type:
   - For GPU support: Select a GPU instance like p3.2xlarge
   - For CPU-only: Select t2.xlarge or larger
6. Configure Instance Details:
   - Use default settings or adjust as needed
7. Add Storage:
   - Increase the size to at least 30GB
8. Configure Security Group:
   - Create a new security group
   - Add rules to allow SSH (port 22) and HTTP (port 80) from your IP
   - Add a custom TCP rule for port 8000 from anywhere (0.0.0.0/0)
9. Review and Launch
10. Create or select an existing key pair and download it
11. Launch the instance

### Step 2: Connect to Your EC2 Instance

```bash
# Make your key pair file read-only
chmod 400 your-key-pair.pem

# Connect to your instance
ssh -i your-key-pair.pem ubuntu@your-instance-public-ip
```

### Step 3: Set Up the Server

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install docker-ce -y

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to the docker group
sudo usermod -aG docker ${USER}
```

Log out and log back in for the group changes to take effect:

```bash
exit
# Then reconnect using the SSH command from Step 2
```

### Step 4: Install NVIDIA Drivers & Docker Runtime (If Using GPU)

```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-525 -y

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 5: Clone Your Repository and Deploy

```bash
# Create a directory for the project
mkdir -p ~/projects
cd ~/projects

# Clone your repository (replace with your actual repository URL)
git clone https://github.com/yourusername/DomainAI.git
cd DomainAI

# Start the application
docker-compose up -d
```

### Step 6: Verify the Deployment

```bash
# Check if containers are running
docker-compose ps

# Check container logs
docker-compose logs

# Test the API locally
curl http://localhost:8000/health
```

### Step 7: Access Your API Publicly

Your API is now accessible using your EC2 instance's public IP address:

```
http://your-ec2-public-ip:8000
```

You can share this URL with others to access your API. For example:
- Health check: `http://your-ec2-public-ip:8000/health`
- API endpoint: `http://your-ec2-public-ip:8000/suggest`

## Option 2: Adding a Domain Name (Optional but Recommended)

### Step 1: Purchase a Domain Name

1. Choose a domain registrar (GoDaddy, Namecheap, Route53, etc.)
2. Purchase your desired domain name (e.g., domainai.com)

### Step 2: Set Up DNS Records

1. Go to your domain registrar's DNS management page
2. Add an A record:
   - Type: A
   - Host: api (or @ for root domain)
   - Value: Your EC2 instance's public IP
   - TTL: 3600 (or default)

### Step 3: Install and Configure Nginx

```bash
# SSH into your EC2 instance
ssh -i your-key-pair.pem ubuntu@your-ec2-public-ip

# Install Nginx
sudo apt install nginx -y

# Create a Nginx configuration file
sudo nano /etc/nginx/sites-available/domainai
```

Add the following configuration (replace yourdomain.com with your actual domain):

```
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/domainai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 4: Set Up HTTPS with Certbot (Recommended)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain SSL certificate
sudo certbot --nginx -d api.yourdomain.com

# Follow the prompts to complete the process
```

### Step 5: Verify Domain Setup

Your API is now accessible using your domain:

```
https://api.yourdomain.com
```

You can share this URL with others to access your API:
- Health check: `https://api.yourdomain.com/health`
- API endpoint: `https://api.yourdomain.com/suggest`

## Making API Requests

Once deployed, you can make requests to your API using:

```bash
curl -X POST https://api.yourdomain.com/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "business_description": "A modern coffee shop specializing in artisanal brews and pastries", 
    "max_tokens": 32
  }'
```

The API will respond with domain name suggestions based on the provided business description.