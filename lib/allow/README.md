### Face based authentication for Linux

It gets tiresome typing password everytime we need root access. This is an attempt to integrate facial recognition into the linux 
authentication process to simplify the process. 

To set it up and running follow the instructions below:

1. Make sure you have run env.sh. Go to the local copy of the repository and fire up the terminal.And type in the following
   command.
   ```
   $ ./env.sh
   ```
   Give root privilages if required.
   This step sets up the conda environment and the required packages.
   

2. cd into the 'allow' folder.
   ```
   $ cd /path/to/local/copy/lib/allow
   ```

3. Now, we have to make allow.sh and allow.py available to PATH. To accomplish this, we copy the files into usr/local/bin
   ```
   $ sudo cp /path/to/local/copy/lib/allow/allow.sh /usr/local/bin
   $ sudo cp /path/to/local/copy/lib/allow/allow.py /usr/local/bin
   ```

4. Assuming that your conda environment is installed, allow.sh can now be used!   
   
   i. Open up a terminal.Type in,
      ```
      $ allow.sh
      ```
      It will prompt you for your root password which it requires for executing sudo commands.
 
 
   ii. Now you gotta 'register' your face.       
      ```
      $ allow.sh -o                                                                                                           
      ```          
      Press Y to finish the onboarding process.
 
 
   iii. Lets run some commands!     
      ```
      $ allow.sh -v <command>                                                                                             
      ```     
      A webcam instance will be started, which will take a capture of the face. If verified, it will run the command as root.
      Any command that can be run by sudo, can be run through this.
      
      
      
#### A note on security
     
This module isn't secure. Its meant to be used as a **convenience tool**. It stores password in a plain txt file. It's meant 
to be used only when the system has a single user. 
Currently working on a much more secure, general version.
