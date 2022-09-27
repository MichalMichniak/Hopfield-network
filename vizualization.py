import pygame as pg
import hopfield
import numpy as np


class MainWindow:
    def __init__(self,hop : hopfield.Hopfield_NN, WIDTH,HEIGH) -> None:
        pg.init()
        pg.display.set_caption("game")
        self.font = pg.font.SysFont(None, 24)
        self.win = pg.display.set_mode((WIDTH*10, HEIGH*10))
        self.hop = hop
        self.WIDTH = WIDTH
        self.HEIGH = HEIGH
        pass

    def run(self):
        """
        run visualization of stabilization hopfield network in time domain
        """
        run = True
        state = np.array([np.random.uniform() - 0.5 for i in range(self.WIDTH*self.HEIGH)])
        #state = np.array([0 for i in range(self.WIDTH*self.HEIGH)])
        # img = self.hop.get_out(np.array([np.random.uniform()*2 - 1 for i in range(self.WIDTH*self.HEIGH)]))[0].reshape([self.WIDTH,self.HEIGH])
        img = state.reshape([self.WIDTH,self.HEIGH])
        ctr = 0
        while run:
            self.win.fill((0,0,0))
            ##### event section #####
            ##### 
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                if event.type == pg.KEYUP:
                    if event.key == pg.K_e:
                        state = np.array([np.random.uniform() - 0.5 for i in range(self.WIDTH*self.HEIGH)])
            
            for n,i in enumerate(img):
                for k,j in enumerate(i):
                    pg.draw.rect(self.win,((0,0,0) if j > 0 else (255,255,255)), (k*10,n*10,10,10))
            if ctr == 2:
                ctr = 0
                state = self.hop.get_next_state(state)
            img = state.reshape([self.WIDTH,self.HEIGH])
            pg.display.update()
            #### delays ####
            pg.time.delay(60)
            ctr +=1
