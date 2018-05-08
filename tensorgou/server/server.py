from SimpleXMLRPCServer import SimpleXMLRPCServer
#from tensorgou.server import train, dtrain, config


def main():
    server = SimpleXMLRPCServer(("10.141.105.108", 8000))
    print "Listening on port 8000..."
    server.register_multicall_functions()
    server.register_function(add, 'add')
    server.register_function(subtract, 'subtract')
    server.register_function(multiply, 'multiply')
    server.register_function(divide, 'divide')
    server.serve_forever()