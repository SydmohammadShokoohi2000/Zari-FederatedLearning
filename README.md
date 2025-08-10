# Zari-FederatedLearning: A Lightweight FL Framework for IoV Intrusion Detection

This repository contains the complete source code and implementation for my Master's thesis project, titled **"A Lightweight Federated Learning Framework for Real-time Intrusion Detection in Vehicular Networks."**

This project is all about tackling a critical challenge: How can we detect malicious attacks within a vehicle's network (like the CAN bus) in real-time, without compromising the privacy of the vehicle's data? Our answer is a custom-built federated learning system.

---
## 🛡️ The Core Idea

Modern vehicles are basically computers on wheels, and just like any computer, they can be hacked. The internal Controller Area Network (CAN bus) that manages everything from the engine to the windows is vulnerable to attacks. We want to build an Intrusion Detection System (IDS) to catch these attacks.

But here's the problem: you can't just send all of a vehicle's driving data to a central server for analysis. That's a huge privacy risk and uses a ton of bandwidth.

This is where Federated Learning (FL) comes in. Here's our workflow:

1.  **No Raw Data Leaves the Vehicle:** Each vehicle (a "client" in our simulation) uses its own raw CAN bus data locally.
2.  **On-Device Pre-processing:** The client transforms sequences of CAN data into image representations. This unique approach turns a time-series data problem into an image classification problem.
3.  **Local Training:** Each client trains a lightweight Convolutional Neural Network (our `ZariTFLite` model) on its own generated images.
4.  **Share Knowledge, Not Data:** Instead of sending data, the client only sends the small, anonymous model updates (the "learnings") to a central server.
5.  **Global Intelligence:** The server aggregates the knowledge from all clients to build a powerful, robust global intrusion detection model, then sends it back to the clients.

This way, we get a smart, collaborative security system where everyone benefits, but no one has to share their private driving data.



---
## 🚀 Key Features

* **Federated Learning:** Implemented using the FedML framework for a cross-silo simulation.
* **CAN-to-Image Conversion:** A custom data pre-processing pipeline that converts raw CAN bus sequences into images, enabling the use of powerful CNNs for intrusion detection.
* **Client-Side Processing:** All data pre-processing and training happens on the edge (the client), which is crucial for privacy and scalability in a real vehicular network.
* **Lightweight CNN Model:** A custom-built `ZariTFLite` model designed to be efficient enough for resource-constrained in-vehicle systems.
* **Simulated IoV Environment:** Uses an MQTT broker to simulate the communication between a central server and multiple vehicle clients in real-time.

---
## ⚙️ Getting Started

Follow these steps to set up and run the project environment.

### Prerequisites

* Python 3.11+
* Git
* An MQTT broker (like Mosquitto) installed and running.
    ```bash
    # On Ubuntu/Debian
    sudo apt-get update && sudo apt-get install mosquitto
    ```

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SydmohammadShokoohi2000/Zari-FederatedLearning.git](https://github.com/SydmohammadShokoohi2000/Zari-FederatedLearning.git)
    cd Zari-FederatedLearning
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv fedml-env
    source fedml-env/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pre-process the Dataset:**
    * Download the Car-Hacking Dataset and place the CSV/txt files into the `data/raw/` directory.
    * Run the pre-processing script. This only needs to be done once. It will create a `processed_data.h5` file in the `data/processed/` directory.
    ```bash
    python dataloader/preprocess_data.py
    ```

---
## 🔧 How to Run the Simulation

To run the full federated learning simulation, you'll need three terminals.

1.  **Start the MQTT Broker:**
    Make sure your Mosquitto service is running. You can check with `sudo systemctl status mosquitto`.

2.  **Start the Server:**
    In your first terminal, start the FedML server. It will initialize and then wait for clients to connect.
    ```bash
    # (Activate the environment first)
    python main.py server --config config/fedml-config.yaml
    ```

3.  **Start the Clients:**
    You'll need a separate terminal for each client.

    * **In Terminal 2 (Client 1):**
        ```bash
        # (Activate the environment first)
        python main.py client --client-id 1 --config config/fedml-config.yaml
        ```
    * **In Terminal 3 (Client 2):**
        ```bash
        # (Activate the environment first)
        python main.py client --client-id 2 --config config/fedml-config.yaml
        ```

Once both clients connect, you'll see the federated learning rounds begin in all three terminals!

---
## 📚 Future Work

This project sets a strong foundation, but there are many exciting avenues for future research:
* **Real Hardware Testing:** Deploying the client-side code on a real embedded device like a Raspberry Pi or NVIDIA Jetson to test performance in a true edge environment.
* **Advanced Aggregation:** Exploring more advanced federated aggregation algorithms beyond FedAvg to handle statistical heterogeneity (Non-IID data).
* **Different Architectures:** Experimenting with other lightweight model architectures like MobileNetV3 or custom attention-based models.

---
## Acknowledgements
A huge thank you to my thesis supervisor, Dr Ehsan Ataei and Dr Yousefpour, and the University of Mazandran,Babolsar,Iran for their support. This work also relies on the "Car-Hacking Dataset" from the Hacking and Countermeasure Research Lab.
