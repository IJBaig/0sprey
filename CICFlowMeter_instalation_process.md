 Install Java 8 (MUST be Java 8, not higher)
sudo apt install openjdk-8-jdk
# If you have multiple Java versions, set Java 8 as default or just see it java 8 is installed and available if you not want do not change the default :
sudo update-alternatives --config java
# Select the java-8 option


Install libpcap
sudo apt-get install libpcap-dev


Install Maven (needed for jnetpcap)
sudo apt install maven


git clone https://github.com/ahlashkari/CICFlowMeter.git
cd CICFlowMeter


# ── LINUX ──────────────────────────────────────────────────
mvn install:install-file \
    -Dfile=jnetpcap/linux/jnetpcap-1.3.0/jnetpcap.jar \
    -DgroupId=org.jnetpcap \
    -DartifactId=jnetpcap \
    -Dversion=1.4.1 \
    -Dpackaging=jar


# Make gradlew executable
chmod +x gradlew


#if you set java 8 as default the run 
./gradlew fatJar
# if java 8 is not default the run
JAVA_HOME=/usr/lib/jvm/temurin-8-jdk-amd64 ./gradlew fatJar
change java-8-openjdk-amd64 to java 8 foldername


ls build/libs/
# CICFlowMeter-4.0-all.jar  ← THIS is standalone, ~17MB
copy both CICFlowMeter-4.0-all.jar and jnetpcap folder from build/libs and CICFlowMeter Directory to src folder


