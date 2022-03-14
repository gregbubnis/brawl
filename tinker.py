#!/usr/bin/env python

import os
import datetime
from datetime import timedelta
import pdb
import glob
import json
import sys

import brawlstats as bs
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#
#
#   TODO: coherent logging
#   TODO: Log stack traces: https://stackoverflow.com/a/5191885/6474403
#   TODO: Ingest json config
#   TODO: Factor data manipulation out of viz
#   TODO: Pull/push PERSISTENT STORAGE
#   TODO: Ingest historical CL data
#   TODO: BrawlApp.run() method rather than cron job
#   NOTE: A bug could arise with CL dates if a late match starts just after the
#         24h CL time block. Precompute CL dates? 
#


def dtobj(x):
    """datetime object for time_raw (BS API format)"""
    return datetime.datetime(int(x[0:4]), int(x[4:6]), int(x[6:8]), int(x[9:11]), int(x[11:13]))

def format_datetime(x):
    """string format"""
    return '%s-%s-%s_%s:%s' % (x[0:4], x[4:6], x[6:8], x[9:11], x[11:13])

def shift_datetime(dt1, delta):
    """shift datetime by delta hours"""
    dtobj = datetime.datetime(int(dt1[0:4]), int(dt1[5:7]), int(dt1[8:10]), int(dt1[11:13]), int(dt1[14:16]))
    dtnew = str(dtobj + timedelta(hours=delta))
    return '%s_%s' % (dtnew[0:10], dtnew[11:16])

def player_battle_log(x, bs_client):
    """DataFrame summary of battle log for a player

    newest results on bottom
    """
    logs = bs_client.get_battle_logs(x)
    player = bs_client.get_player(x)
    data = []
    for battle in logs:
        dd = dict(
            datetime=str(format_datetime(battle['battle_time'])),
            date=str(format_datetime(battle['battle_time'])[0:10]),
            time_raw=str(battle['battle_time']),
            mode=str(battle['event'].get('mode', 'none')),
            type=str(battle['battle'].get('type', 'none')),
            result=str(battle['battle'].get('result', 'none')),
            map=str(battle['event']['map']),
            rank=str(battle['battle'].get('rank', 'none')),
            trophy_change=str(battle['battle'].get('trophy_change', 'none')),
        )
        data.append(dd)
    df = pd.DataFrame(data)
    df['name'] = [player.name]*len(df)
    df['tag'] = [player.tag]*len(df)
    df['override'] = [False]*len(df)
    df = add_isCL_col(df)
    df = df.sort_values(by='datetime', ascending=True).reset_index(drop=True)
    return df

def fix_time_cols(df):
    """post hoc renaming"""
    if 'time' in df.columns:
        df.rename(columns={'time':'datetime'}, inplace=True)
    return df

def add_isCL_col(df):
    """add the 'is_CL' column to a DataFrame (inplace)"""
    if 'is_CL' not in df.columns:
        bt = df['type']
        tc = df['trophy_change']
        df['is_CL'] = ((bt == 'teamRanked') | (bt == 'soloRanked')) & (tc != 'none')
    return df


class BrawlApp(object):
    """"""
    def __init__(self, cfg):
        if cfg:
            with open(cfg) as jfopen:
                jdict = json.load(jfopen)
        else:
            jdict = dict()

        defaults = dict(
            output_folder='./brawlapp',
            club_token="2R822GG9V",
            api_token=None
        )

        #self.club_token = "2R822GG9V"

        api_token = jdict.get('api_token', defaults['api_token'])
        club_token = jdict.get('club_token', defaults['club_token'])
        self.output_folder = jdict.get('output_folder', defaults['output_folder'])
        self.battles_folder = os.path.join(self.output_folder, 'battles')
        os.makedirs(self.battles_folder, exist_ok=True)

        # python API client
        self.bs_client = bs.Client(api_token)
        self.club = self.bs_client.get_club(club_token)
        self.make_roster()

        print('==== roster 1 ====')
        print(self.df_roster)

        print(str(datetime.datetime.utcnow())[:19]+' SUCCESS __init__')


    def fetch_API_data(self):
        """fetch battle logs and update the csv log files"""
        members = self.club.get_members()

        # make a dataframe of recent battles for current members
        battle_data = []
        for m in members:
            try:
                df_battles = player_battle_log(m.tag, self.bs_client)
                battle_data.append(df_battles)
            except Exception as e:
                print('  '+str(e).replace('\n'," "))
                print('       could not get logs for %s %s' % (m.name, m.tag))
        # most recent battles
        df_battles = pd.concat(battle_data, axis=0).reset_index(drop=True)

        # Fuse df_battles to the existing battle log files and remove duplicate
        # entries (could be more elegant)
        dates = sorted(df_battles['date'].unique())
        all_battles = []
        for dd in dates:
            logfile = os.path.join(self.battles_folder, 'battles-%s.csv' % dd)
            dfd = df_battles[df_battles['date'] == dd]
            if os.path.isfile(logfile):
                #print('log file (%s) exists' % logfile)
                dfd_old = pd.read_csv(logfile, index_col=0)
                dfd = pd.concat([dfd_old, dfd], axis=0).drop_duplicates(ignore_index=True).sort_values(by='datetime', ascending=True).reset_index(drop=True)
                # Hacky mechanism to override erroneous API data!
                #
                # Edit a line of the csv, set 'override' == True, and then
                # save it.
                # On the next run of the program, the edited row will replace
                # the API fetched row. The matching of rows is based on the
                # 'timeraw' and 'tag' columns, so do not edit those
                #
                # find override rows
                dfd_over = dfd[dfd['override']]
                if len(dfd_over) > 0:
                    drop_ix = []
                    # check if each row of dfd should be overridden
                    for ixx, row in dfd.iterrows():
                        for ixxx, roww in dfd_over.iterrows():
                            if row['time_raw'] == roww['time_raw'] and row['tag'] == roww['tag'] and row['override']==False:
                                print('----')
                                print('DROP IT\n', ixx, dict(row))
                                print('KEEP IT\n', ixxx, dict(roww))
                                drop_ix.append(ixx)
                    dfd = dfd.drop(drop_ix, axis=0).reset_index(drop=True)
                dfd.to_csv(logfile)
            else:
                #print('log file (%s) DNE' % logfile)
                dfd.to_csv(logfile)
            all_battles.append(dfd)
        self.df_battles = pd.concat(all_battles).reset_index(drop=True)
        print(str(datetime.datetime.utcnow())[:19]+' SUCCESS fetch_API_data')

    # def add_override_col(self):
    #     """one off"""
    #     logs = sorted(glob.glob('%s/battles*csv' % self.battles_folder))
    #     for log in logs:
    #         print('adding override col to %s' % log)
    #         df = pd.read_csv(log, index_col=0)
    #         df['override']=[False]*len(df)
    #         df.to_csv(log)

    def make_roster(self, tags=None):
        # current roster + extra tags (e.g. for former members)
        active_tags = [m.tag for m in self.club.get_members()]
        if isinstance(tags, list):
            all_tags = active_tags + tags
        elif isinstance(tags, str):
            all_tags = active_tags + [tags]
        else:
            all_tags = active_tags
        members = [self.bs_client.get_player(x) for x in all_tags]

        # build dataframe
        member_data = [dict(name=m.name, tag=m.tag, trophies=m.trophies, current=m.tag in active_tags) for m in members]
        self.df_roster = pd.DataFrame(member_data).drop_duplicates(ignore_index=True).sort_values(by=['current', 'trophies'], ascending=[False, False])


    def process_data(self):
        """"""

        # concatenate ALL the logs
        logs = sorted(glob.glob('%s/battles*csv' % self.battles_folder))
        big_df = pd.concat([pd.read_csv(x, index_col=0) for x in logs], axis=0, ignore_index=True).sort_values(by='datetime', ascending=True).reset_index(drop=True)
        df_battles = big_df
        self.df_battles = df_battles

        # Update roster with all members (active or kicked) in the logs
        self.make_roster(tags=list(big_df['tag'].unique()))
        df_members = self.df_roster.sort_values(by=['current', 'trophies'], ascending=[True, True]).reset_index(drop=True)
        self.df_members = df_members

        print('==== roster 2 ====')
        print(self.df_roster)

        # dataframe just of CL values
        df_cl = df_battles[df_battles['is_CL']].copy()

        # generate club league dates
        day1 = datetime.datetime(2022, 2, 3, 12, 0)
        day2 = datetime.datetime(2022, 2, 5, 12, 0)
        day3 = datetime.datetime(2022, 2, 7, 12, 0)
        daynow = datetime.datetime.utcnow()
        cl_dates = [day1, day2, day3]
        while (cl_dates[-1]<=daynow):
            cl_dates += [x + timedelta(days=7) for x in cl_dates[-3:]]

        # map cl battles to the closest cl date
        # (robust to the case where a very late battle spills over past 24h)
        cl_dates_arr = np.array([x.timestamp()/86400 for x in cl_dates])
        cl_battles_arr = [dtobj(x).timestamp()/86400 for x in df_cl['time_raw']]
        ix = [np.ndarray.argmin(np.abs(cl_dates_arr-x)) for x in cl_battles_arr]
        df_cl['date_CL'] = [str(cl_dates[i])[0:10] for i in ix]
        df_cl = df_cl.astype({'trophy_change':int})
        self.df_cl = df_cl

    def viz(self):
        """quick and dirty matplotlib visualization to png file"""
        num_battles = 50


        df_battles = self.df_battles
        df_cl = self.df_cl
        df_members = self.df_members


        # moar dataframes
        df_plt = self.df_cl[['date_CL', 'tag', 'trophy_change']]
        df_summed = df_plt.groupby(['date_CL', 'tag'], as_index=False).sum()

        # get CL daily avgs
        tag2name= dict(zip(df_members['tag'], df_members['name']))
        df_avgs = df_summed.groupby(['tag'], as_index=False).mean()
        df_avgs['name'] = [tag2name[x] for x in df_avgs['tag']]


        # # the figure
        # fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(8, 12)) #, gridspec_kw=gs_kw)
        # # PLOT RECENT BATTLE RESULTS (PANEL 0)
        # # making a discrete colormap
        # clist = ['tab:red', 'lightgrey', 'tab:green']
        # bounds = np.linspace(-1.2, 1.2, 4)
        # norm = mpl.colors.BoundaryNorm(bounds, 3)
        # cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', clist, 3)
        # zdict = dict(victory=1, defeat=-1, draw=0, none=0)
        # im = np.zeros((len(df_members), num_battles))
        # for ix, row in df_members.iterrows():
        #     tag = row['tag']
        #     bdata = df_battles[df_battles['tag'] == tag].sort_values(by='datetime', ascending=True)[-num_battles:].reset_index(drop=True)
        #     pad = num_battles-len(bdata)
        #     #print('name %25s  pad %i' % (row['name'], pad))
        #     # plot CL trophies
        #     for ixx, battle in bdata.iterrows():
        #         if battle['is_CL']:
        #             axs[1].text(ixx+pad, ix, str(battle['trophy_change']), zorder=3, ha='center', va='center')
        #     # win loss draw rasters
        #     res = ['none']*pad+bdata['result'].values.tolist()
        #     im[ix, :] = [zdict[x] for x in res]
        # now_utc = str(datetime.datetime.utcnow())
        # title = 'most recent battle results\n'
        # title += '%s %s UTC' % (now_utc[:10], now_utc[11:19])
        # axs[1].imshow(im, cmap=cmap, norm=norm, alpha=0.5)
        # axs[1].set_xticks([0, num_battles-1], ['old', 'new'])
        # axs[1].set_yticks(range(len(df_members)), df_members['name'])
        # axs[1].set_title(title)
        # # export png
        # plt.tight_layout()
        # out_file = os.path.join(self.output_folder, 'plot-recent-battles.png')
        # plt.savefig(out_file)

        # NEXT FIGURE 7 DAYS OF BATTLES

        cl_dates = sorted(self.df_cl['date_CL'].unique())

        gs_kw = dict(height_ratios=[1, 1, 1])
        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(8, 22), gridspec_kw=gs_kw)
        now_utc = datetime.datetime.utcnow()

        for ix, row in df_members.iterrows():
            bdata = df_battles[df_battles['tag'] == row['tag']].sort_values(by='datetime', ascending=True).reset_index(drop=True)
            datetimes = [dtobj(x) for x in bdata['time_raw']]

            # only keep the last 7 days
            dt7daysago = now_utc-timedelta(days=7)
            ixkeep = [x>dt7daysago for x in datetimes]
            bdata = bdata[ixkeep].reset_index(drop=True)
            datetimes = [dtobj(x) for x in bdata['time_raw']]

            # plot CL trophies (offset if adjacent numbers are too close)
            xyp = []
            for ixx, battle in bdata.iterrows():
                if battle['is_CL']:
                    xyp.append([datetimes[ixx], ix+0.25, str(battle['trophy_change'])])
                    if len(xyp)>1:
                        diff = (xyp[-1][0]-xyp[-2][0]).total_seconds()/60
                        if diff < 60:
                            xyp[-1][0]+=timedelta(hours=2-(diff/60))
                    axs[0].text(xyp[-1][0], xyp[-1][1], xyp[-1][2], zorder=3, ha='center', va='center', color='tab:red', fontsize='small')

            # plot all battles as grey dots
            axs[0].scatter(datetimes, [ix]*len(bdata), marker='o', alpha=0.3, color='grey', s=20)
            if ix%2 == 0:
                axs[0].axhline(ix, color='grey', alpha=0.2, lw=1)

        axs[0].yaxis.set_tick_params(size=0)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_yticks(range(len(df_members)), df_members['name'])
        plt.draw()
        #axs[0].set_xticks(axs[0].get_xticks(), rotation=55, ha='right')
        #axs[0].set_xticks(axs[0].get_xticks(), axs[0].get_xticklabels(), rotation=55, ha='right')
        #axs[0].tick_params(axis='x', labelrotation=55) #, ha='right')

        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%m/%d')
        axs[0].xaxis.set_major_formatter(myFmt)
        axs[0].axvline(datetime.datetime.utcnow(), color='grey', alpha=0.5, lw=1)
        title = '7 days of activity\n'
        title += '%s %s UTC' % (str(now_utc)[:10], str(now_utc)[11:19])
        axs[0].set_title(title)

        print(df_members['current'].value_counts().to_dict())
        num_former = df_members['current'].value_counts().to_dict()[False]

        axs[1].sharey(axs[0])
        #cl_dates = sorted(df_cl['date_CL'].unique())


        # print(df_summed.groupby(['date_CL'], as_index=False).sum())
        # print(df_summed.columns)
        # print(df_summed.head(50))


        colmap = dict(zip(cl_dates , range(len(cl_dates))))
        rowmap = dict(zip(df_members['tag'], range(len(df_members))))
        df_summed['col'] = [colmap[x] for x in df_summed['date_CL']]
        df_summed['row'] = [rowmap[x] for x in df_summed['tag']]

        arr = np.zeros((len(df_members), len(cl_dates))).astype(int)
        for ix, row in df_summed.iterrows():
            tc = row.get('trophy_change', 0)
            arr[row['row'], row['col']] = tc

            if tc<10:
                color='w'
            else:
                color='k'
            axs[1].text(row['col'], row['row'], str(tc), ha='center', va='center', color=color)

            from matplotlib.lines import Line2D

            if ix%2 == 0 and ix<len(df_members):
                #xxx = axs[1].plot([-3, -1], [ix, ix], '-', color='grey', )
                xxx = Line2D([-1, -0.5], [ix, ix], ls='-', color='grey', alpha=0.2, lw=1)
                axs[1].add_line(xxx)
                xxx.set_clip_on(False)
            #axs[1].set_clip_on(False)



        avgdict = dict(zip(df_avgs['tag'], df_avgs['trophy_change']))
        avgs = [avgdict.get(k, 0) for k in df_members['tag']]
        #print(len(avgs), len(df_members))
        for ix, row in df_members.iterrows():
            #axs[1].text((ix-0.5)/len(df_members), 0.9, str(row['trophy_change']), ha='center', va='center', color='grey', transform=plt.gcf().transFigure)
            axs[1].text(len(cl_dates)+0.25, ix, '%3.1f' % avgs[ix], ha='center', va='center', color='k')
            #print('%20s' % row['name'], row['trophy_change'])
        #plt.subplots_adjust(right=0.25)


        axs[1].imshow(arr, aspect=0.5, cmap='bone', vmin=-2, vmax=27, origin='lower')
        #axs[1].set_yticks(range(len(df_members)), df_members['name'])
        #axs[1].set_yticklabels([])
        #plt.setp(axs[1].get_yticklabels(), visible=False)
        #axs[1].yaxis.set_tick_params(size=0)

        axs[1].axhline(num_former-0.5, color='salmon')

        #plt.setp(axs[1].get_yticklabels(), visible=False)
        # plt.tight_layout(h_pad=-3)
        axs[1].set_xticks(range(len(cl_dates)), [x[-5:] for x in cl_dates], rotation=70, ha='center')
        axs[1].set_title('club league trophies (avg excluding zero days)')


        ## CDF
        df_avgs_sorted = df_avgs.sort_values(by='trophy_change', ascending=True).reset_index(drop=True)

        print(df_avgs_sorted)
        axs[2].plot(df_avgs_sorted['trophy_change'], df_avgs_sorted.index, '-o')
        #axs[2].set_aspect(1.)
        axs[2].grid()

        #axs[1].set_yticks(axs[1].get_yticks()[::2])
        
        #plt.subplots_adjust(wspace=-2.0)
        #plt.tight_layout(h_pad=0)
        plt.tight_layout(h_pad=4)

        #plt.subplots_adjust(bottom=0.15)


        #plt.tight_layout(h_pad=-3)
        out_file = os.path.join(self.output_folder, 'plot-7day-battles.png')
        plt.savefig(out_file)



        print(str(datetime.datetime.utcnow())[:19]+' SUCCESS viz')




    def pull_stored_data(self):
        pass
    def push_stored_data(self):
        pass
    def run(self):
        pass

if __name__ == "__main__":

    try:
        cfg = sys.argv[1]
    except:
        cfg = None
    xx = BrawlApp(cfg)
    xx.fetch_API_data()
    xx.process_data()
    xx.viz()
