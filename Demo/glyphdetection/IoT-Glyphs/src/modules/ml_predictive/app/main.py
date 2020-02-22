import os
import random
import sys
import time
import numpy as np
import json
import tensorflow as tf
import iothub_client
from inference import predict
from iothub_client import (IoTHubModuleClient, IoTHubClientError, IoTHubError,
                           IoTHubMessage, IoTHubMessageDispositionResult,
                           IoTHubTransportProvider)

SEND_CALLBACKS = 0

def receive_message(message, hubManager):
    results = predict()
    print(results)
    send_to_Hub_callback(json.dumps(results))
    return IoTHubMessageDispositionResult.ACCEPTED

def send_to_Hub_callback(strMessage):
    print(strMessage)
    message = IoTHubMessage(bytearray(strMessage, 'utf8'))
    hubManager.send_event_to_output("output", message, 0)

def send_confirmation_callback(message, result, user_context):
    global SEND_CALLBACKS
    SEND_CALLBACKS += 1    

class HubManager(object):

    def __init__(
            self,
            messageTimeout,
            protocol,
            verbose=False):
        '''
        Communicate with the Edge Hub

        :param int messageTimeout: the maximum time in milliseconds until a message times out. The timeout period starts at IoTHubClient.send_event_async. By default, messages do not expire.
        :param IoTHubTransportProvider protocol: Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
        :param bool verbose: set to true to get detailed logs on messages
        '''
        self.messageTimeout = messageTimeout
        self.client_protocol = protocol
        self.client = IoTHubModuleClient()
        self.client.create_from_environment(protocol)
        self.client.set_option("messageTimeout", self.messageTimeout)
        self.client.set_option("product_info","edge-engine-inference")
        if verbose:
            self.client.set_option("logtrace", 1)
        self.client.set_message_callback("output", receive_message,self)

    def send_event_to_output(self, outputQueueName, event, send_context):
        self.client.send_event_async(outputQueueName, event, send_confirmation_callback, send_context)


def main():

    try:
        print ( "\nPython %s\n" % sys.version )
        print ( "Inference Module Sensors Azure IoT Edge Module. Press Ctrl-C to exit." )
        try:
            global hubManager
            hubManager = HubManager(10000, IoTHubTransportProvider.MQTT)
            while True:
                time.sleep(1000)

        except IoTHubError as iothub_error:
            print ( "Unexpected error %s from IoTHub" % iothub_error )
            return        

    except KeyboardInterrupt:
        print ( "Inference engine module stopped" )


if __name__ == '__main__':    
    main()
