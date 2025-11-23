from robot import *

leader, follower = connect_leader, connect_follower()
while True:
    follower.send_action(leader.get_observation)