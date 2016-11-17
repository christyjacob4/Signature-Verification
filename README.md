# Signature-Verification

A Simple Python app that helps detect if a signature is real or forged.
Supply samples of your original signature which is used to train the system and then, supply the signature to be tested.
The training samples are required to be named sequentially as "sign0.jpg" , "sign1.jpg" ,"sign2.jpg" ,.... and so on. This can be changed in the code according to your liking by modifying the following statement from the code:
#####
for j in range(19):
     image_sources.append("sign"+str(j)+".jpg")
####    

The test sample is named forged.jpg but you will be asked for the filename once you enter the program.

The training samples must be exposed with proper focus on the signature and minimal noise.
It is still under development and I'm trying to fully automate this process.


Requirements:
Python 2.7
Open CV
Imutils


