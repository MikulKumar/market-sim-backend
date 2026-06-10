#Agent-Based Financial Market Simulation

import heapq #prioity, gives best option next

import time

import random

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import inspect
# order book

class Order:
    def __init__(self, order_id, side, order_type, price, quantity):
        self.order_id = order_id
        self.side = side
        self.order_type = order_type
        self.price = float(price)
        self.quantity = int(quantity)
        self.Timestamp = time.time()
        

#OrderBook

class OrderBook:
    def __init__(self, start_price):
        self.bids = {}
        self.asks = {}
        self.order_map = {}
        self.trades = []
        self.cancelled = set()
        self.start_price = start_price

    def cancel(self, order_id):
      self.cancelled.add(order_id)

    def best_bids(self):
        return max(self.bids.keys()) if self.bids else None

    def best_asks(self):
        return min(self.asks.keys()) if self.asks else None

    def rest(self, order):
        if order.side == 'buy':
            if order.price not in self.bids:
              self.bids[order.price] = []
            self.bids[order.price].append(order)
        elif order.side == 'sell':
            if order.price not in self.asks:
              self.asks[order.price] = []
            self.asks[order.price].append(order)

        #used to locate the heap/order
        self.order_map[order.order_id] = order

    def printtable(self):
      '''
      returns the bids and asks
      '''
      return self.bids , self.asks

    #match function

    def match(self, order):
        if order.side == 'buy':
          while order.quantity > 0 and self.asks:
            best_price = min(self.asks.keys())

            if order.order_type == 'limit' and best_price > order.price:
              break

            queue = self.asks[best_price]

            while queue and queue[0].order_id in self.cancelled:
              queue.pop(0)

            if not queue:
              del self.asks[best_price]
              continue

            best_order = queue[0]
            trade_qty = min(order.quantity, best_order.quantity)

            self.trades.append({
              'price': best_price,
              'quantity': trade_qty
            })

            order.quantity -= trade_qty
            best_order.quantity -= trade_qty

            if best_order.quantity == 0:
              queue.pop(0)

            if not queue:
              del self.asks[best_price]

        elif order.side == 'sell':
          while order.quantity > 0 and self.bids:
            best_price = max(self.bids.keys())

            if order.order_type == 'limit' and best_price < order.price:
              break

            queue = self.bids[best_price]

            while queue and queue[0].order_id in self.cancelled:
              queue.pop(0)

            if not queue:
              del self.bids[best_price]
              continue

            best_order = queue[0]
            trade_qty = min(order.quantity, best_order.quantity)

            self.trades.append({
              'price': best_price,
              'quantity': trade_qty
            })

            order.quantity -= trade_qty
            best_order.quantity -= trade_qty

            if best_order.quantity == 0:
              queue.pop(0)

            if not queue:
              del self.bids[best_price]


    def submit(self, order):
      if order.order_type == "market":
        self.match(order)
      elif order.order_type == "limit":
        self.match(order)         #try to match it first
        if order.quantity > 0:    #if leftover rest it in the book
          self.rest(order)


    def purge_cancelled(self):
      for price in list(self.bids.keys()):
        self.bids[price] = [o for o in self.bids[price]
                            if o.order_id not in self.cancelled]
        if not self.bids[price]:
          del self.bids[price]

      for price in list(self.asks.keys()):
        self.asks[price] = [o for o in self.asks[price]
                            if o.order_id not in self.cancelled]
        if not self.asks[price]:
          del self.asks[price]

      self.cancelled.clear()


#All agents

class Agent:
  def __init__(self, order_book, agent_id):
    self.book = order_book
    self.agent_id = agent_id
    self.order_counter = 0


  def next_id(self):
    self.order_counter += 1
    return f"{self.agent_id}_{self.order_counter}"

  def act(self):
    raise NotImplementedError

  def mid_price(self):
    b = self.book.best_bids()
    a = self.book.best_asks()
    if b and a:
      return (a + b) / 2
    elif self.book.trades:
      return self.book.trades[-1]['price']
    return None


#Noise trader

class NoiseTrader(Agent):
  def __init__(self,order_book, agent_id,min_qty=0,max_qty=1500,buy_bias=0.5):
    super().__init__(order_book, agent_id)
    self.max_qty = max_qty
    self.buy_bias = buy_bias
    self.min_qty = min_qty
  def act(self):
    side = 'buy' if random.random() < self.buy_bias else 'sell'
    qty = max(1, random.randint(self.min_qty, max(self.min_qty, self.max_qty)))

    self.book.submit(Order(self.next_id(), side, 'market', 0, qty))


#market Maker

class MarketMaker(Agent):
  def __init__(self, order_book, agent_id,inventory_factor=4, skew_factor=0.00001,start_price=100,qty=1500,spread=1.0, aggression=0.5):
    super().__init__(order_book,agent_id)
    self.spread = spread
    self.start_price = start_price
    self.qty = qty
    self.aggression = aggression
    self.inventory = 0
    self.active_order_ids = {}
    self.skew_factor = skew_factor
    self.inventory_factor = inventory_factor

  def act(self):
    for order_id, (original_qty, side) in self.active_order_ids.items():
      order = self.book.order_map.get(order_id)
      filled = original_qty - order.quantity if order else original_qty
      self.inventory += filled if side == 'buy' else -filled
      
    for order_id, (original_qty, side) in self.active_order_ids.items():
      self.book.cancel(order_id)  #adds the order_id to the cancelled set

    self.active_order_ids = {} #resets the order_id to null
    
    self.book.purge_cancelled()   #orders cancelled here

    mid = self.mid_price()
    if mid is None:
      mid = self.book.start_price

    skewed_mid = max(1, mid - (self.inventory * self.skew_factor))

    bid_price = round(skewed_mid - self.spread / 2, 2)
    ask_price = round(skewed_mid + self.spread / 2, 2)

    bid_qty = int(self.qty * random.uniform(0.7, 1.3))
    ask_qty = int(self.qty * random.uniform(0.7, 1.3))
    max_inventory = self.qty * self.inventory_factor

    if random.random() < self.aggression or len(self.book.trades) == 0:

        if self.inventory < -max_inventory:
          bid = Order(self.next_id(), 'buy', 'limit', bid_price, bid_qty)
          self.book.submit(bid)
          self.active_order_ids[bid.order_id] = (bid_qty,"buy")
        elif self.inventory > max_inventory:
          ask = Order(self.next_id(), 'sell', 'limit', ask_price, ask_qty)
          self.book.submit(ask)
          self.active_order_ids[ask.order_id] = (ask_qty,"sell")
        else:
          bid = Order(self.next_id(), 'buy', 'limit', bid_price, bid_qty)
          ask = Order(self.next_id(), 'sell', 'limit', ask_price, ask_qty)

          self.book.submit(bid)
          self.book.submit(ask)

          self.active_order_ids[bid.order_id] = (bid_qty,"buy")
          self.active_order_ids[ask.order_id] = (ask_qty,"sell")
        

#Trend Follower

class Trendfollower(Agent):
  def __init__(self,order_book, agent_id, lookback=10, qty=60, aggression=0.5):
    super().__init__(order_book, agent_id)
    self.lookback = lookback
    self.qty = qty
    self.aggression = aggression

  def act(self):
    if len(self.book.trades) < self.lookback:
      return

    recent = [t['price'] for t in self.book.trades[-self.lookback:]]

    if recent[-1] > recent[0]:
      side = 'buy'
    elif recent[-1] < recent[0]:
      side = 'sell'
    else:
      return

    if random.random() < self.aggression:
      self.book.submit(Order(self.next_id(), side, 'market', 0, self.qty))


#Whale

class Whale(Agent):
  def __init__(self,order_book, agent_id, max_qty=2000):
    super().__init__(order_book, agent_id)
    self.max_qty = max_qty

  def act(self):
    if len(self.book.trades) == 0:
        return

    side = random.choice(['buy','sell'])
    qty= random.randint(500,self.max_qty)

    if random.random() < 0.05:  # 5% chance per tick
      self.book.submit(Order(self.next_id(), side, 'market', 0, qty))




#Retail trader

class RetailTrader(Agent):
  def __init__(self,order_book, agent_id,cash=10000,take_profit=0.05, stop_loss=0.3):
    super().__init__(order_book, agent_id)
    self.start_cash = cash
    self.cash = cash
    self.position = 0
    self.avg_entry = 0
    self.take_profit = take_profit
    self.stop_loss = stop_loss

  def act(self):
    mid = self.mid_price()
    if mid is None:
      return

    if self.position > 0:
      change = (mid - self.avg_entry) / self.avg_entry

      if change >= self.take_profit:
        self.book.submit(Order(self.next_id(), 'sell', 'market', 0, self.position))
        self.cash += self.position * mid
        self.position = 0


      elif change <= -self.stop_loss:
        self.book.submit(Order(self.next_id(), 'sell', 'market', 0, self.position))
        self.cash += self.position * mid
        self.position = 0

    if self.cash > 0:
      if random.random() < 0.02:
        qty = int(self.cash * 0.1 / mid)
        if qty > 0:
          self.book.submit(Order(self.next_id(), 'buy', 'market', 0, qty))
          self.avg_entry = mid
          self.position = qty

  def pnl(self):
        mid = self.mid_price()
        if mid is None:
            return 0
        # cash + value of current position - starting cash
        return (self.cash + self.position * mid) - self.start_cash


#Mean Reversion

class MeanReversion(Agent):
  def __init__(self, order_book, agent_id, mean_window=30, qty=200, aggression=0.5):
    super().__init__(order_book, agent_id)
    self.mean_window = mean_window
    self.aggression = aggression
    self.qty = qty

  def act(self):
    if len(self.book.trades) < self.mean_window:
      return
    if random.random() < self.aggression:
      if self.book.trades[-self.mean_window]['price'] < self.book.trades[-1]['price']:
        self.book.submit(Order(self.next_id(), 'sell', 'market', 0, self.qty))
      elif self.book.trades[-self.mean_window]['price'] > self.book.trades[-1]['price']:
        self.book.submit(Order(self.next_id(), 'buy', 'market', 0, self.qty))
      else:
        return


#Order flow trader

class OrderFlow(Agent):
  def __init__(self, order_book, agent_id, threshold = 0, window=10,qty=400, aggression=0.5):
    super().__init__(order_book, agent_id)
    self.threshold = threshold
    self.aggression = aggression
    self.qty = qty
    self.window = window
    self.imbalance_history = []

  def act(self):
    bid_volume = sum(o.quantity for queue in self.book.bids.values()
                    for o in queue if o.order_id not in self.book.cancelled)
    ask_volume = sum(o.quantity for queue in self.book.asks.values()
                    for o in queue if o.order_id not in self.book.cancelled)

    if (bid_volume + ask_volume) == 0:
      return

    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    #print("ask_volume",ask_volume)
    #print("bid_volume",bid_volume)
    #print('imbalance',imbalance)

    self.imbalance_history.append(imbalance)
    if len(self.imbalance_history) > self.window:
      self.imbalance_history.pop(0)  # drop oldest

    imbalance_mean = sum(self.imbalance_history) / len(self.imbalance_history)


    if imbalance_mean > self.threshold:
      self.book.submit(Order(self.next_id(), 'buy', 'market', 0, self.qty))
    elif imbalance_mean < -self.threshold:
      self.book.submit(Order(self.next_id(), 'sell', 'market', 0, self.qty))
    else:
      return

#Simulation loop

class simulation:
  def __init__(self,start_price=100.0):
    self.book = OrderBook(start_price=start_price)
    self.agents = []
    self.order_count = 0
    self.start_price = start_price

  def add_agent(self,agent):
    self.agents.append(agent)

  def run(self, ticks=500):
    for tick in range(ticks):
        current_price = self.book.trades[-1]['price'] if self.book.trades else self.book.start_price
        if current_price <= 0:
            print("Stock reached zero, halting simulation")
            break
        for agent in self.agents:
            agent.act()

  def price_series(self):
    return [t['price'] for t in self.book.trades]

  def volume_series(self):
    return [t['quantity'] for t in self.book.trades]

  def describe(self,agent_class):
    return inspect.signature(agent_class.__init__).parameters


  def add_population(self, agent_class, num, params):   #num, qty_mean, qty_std
    rng = np.random.default_rng()

    for x in range(num):
      drawn_params={}

      for key,values in params.items():
        if isinstance(values, tuple):
          min_val = values[3] if len(values) > 3 else 1
          max_val = values[4] if len(values) > 4 else None

          if values[2] > 0:
            value_1 = values[0]
            value_2 = values[1]
            drawn_params[key]= round(float(np.clip(rng.normal(loc=value_1, scale=value_2), min_val, max_val)),values[2]) #loc=mean, scale=std

          else:
            drawn_params[key]= int(np.clip(round(rng.normal(loc=values[0], scale=values[1]),values[2]), min_val, max_val)) #loc=mean, scale=std
        else:
          drawn_params[key] = values

      self.order_count += 1

      agent_name = agent_class.__name__

      self.add_agent(agent_class(self.book, agent_id=f"{agent_name}_{self.order_count}",**drawn_params))

      print(agent_class, drawn_params)
