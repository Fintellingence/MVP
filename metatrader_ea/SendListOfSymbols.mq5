//+------------------------------------------------------------------+
//|                                             test_exportation.mq5 |
//|                        Copyright 2020, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2020, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"



void wait_period(uint time_milisec)
{
   uint t = GetTickCount() + time_milisec;
   while (true)
   {
      if (GetTickCount() > t) break;
   }
}



bool socksend(int sock, string request)
{
   char req[];
   int len = StringToCharArray(request,req)-1;
   if (len < 0) return(false);
   return (SocketSend(sock,req,len) == len);
}



string socketreceive(int sock,int timeout)
{
   char rsp[];
   string result = "";
   uint len;
   uint timeout_check = GetTickCount() + timeout;
   do
   {
      len = SocketIsReadable(sock);
      if(len)
      {
         int rsp_len;
         rsp_len = SocketRead(sock,rsp,len,timeout);
         if(rsp_len > 0)
         {
            result += CharArrayToString(rsp,0,rsp_len); 
         }
      }
   }
   while((GetTickCount() < timeout_check) && !IsStopped());

   return result;
}



uint AssertSymbols()
{

   int
      socket;
   bool
      isCustom;
   uint
      n_sec,
      Nsymbols;
   ushort
      sep_code;
   string
      sep_str,
      symbols,
      valid_symbols,
      symbols_split[];

   isCustom = true;
   Nsymbols = 0;
   valid_symbols = "";
   sep_str = "_"; // The separator must match the one defined in python
   sep_code = StringGetCharacter(sep_str,0);
   
   socket = SocketCreate();
   if (socket == INVALID_HANDLE)
   {
      Print("Invalid socket created in function AssertSymbols");
      return 0;
   }

   if(SocketConnect(socket,"localhost",9090,10))
   {
      Print("Waiting for list of symbols from python server ... ");
      symbols = socketreceive(socket,10);
      n_sec = 0;
      while (StringLen(symbols) == 0)
      {
         wait_period(1000);   // force to wait 1 sec until next read
         symbols = socketreceive(socket,10);
         n_sec++;
         if (n_sec > 60)
         {
            Print("Error. Could not receive any symbol in acceptable time.");
            SocketClose(socket);
            return 0;
         }
      }
      Print("List of symbols successfully received.");
      StringSplit(symbols,sep_code,symbols_split);
      for (int i = 0; i < ArraySize(symbols_split); i++)
      {
         // include only valid symbols to send back to server
         if (SymbolExist(symbols_split[i],isCustom))
         {
            if (Nsymbols == 0) valid_symbols += symbols_split[i];
            else               valid_symbols += "_" + symbols_split[i];
            Nsymbols += 1;
            Print(symbols_split[i]);
         }
         else Print(symbols_split[i]," -> Not found");
      }
      // send back only valid symbols separated by underscores _
      string received = socksend(socket,valid_symbols) ? socketreceive(socket,10) : "";
   }
   else
   {
      Print("Connection ","localhost",":",9090," error ",GetLastError());
      Print("Error occurred in function to assert symbols.");
      Print("Assert python server were running!");
   }
   SocketClose(socket);
   return Nsymbols;
}



bool SendOHLCData()
{

   bool
      isCustom;
   int
      socket,
      n_sec,
      i,
      cp_time,
      cp_open,
      cp_high,
      cp_low,
      cp_close,
      cp_tickvol,
      cp_vol;
   ushort
      sep_code;      // string separator code
   string
      symbol,
      to_send,       // output msg to send to python function
      inp_time_msg,  // input msg get from python function
      sep_str,       // string with the separator character
      msg_split[];   // input message after split
   long
      vol[1440],
      tick_vol[1440];
   double
      open_price[1440],
      high_price[1440],
      low_price[1440],
      close_price[1440];
   datetime
      d1,
      d2,
      DayEnd,
      time_arr[1440];
   MqlDateTime
      DayEnd_strc;

   sep_str = "_";    // The separator must match the one defined in python
   sep_code = StringGetCharacter(sep_str,0);
   isCustom = true;  // needed to check if SymbolExist
   
   socket = SocketCreate();
   if (socket == INVALID_HANDLE)
   {
      Print("Invalid socket created in function SendOHLCData");
      return false;
   }

   if(SocketConnect(socket,"localhost",9090,10))
   {
      Print("Connected to ","localhost",":",9090);
      symbol = socketreceive(socket,10);  // receive company symbol from python
      n_sec = 0;
      while (StringLen(symbol) == 0)
      {
         wait_period(1000);
         symbol = socketreceive(socket,10);
         n_sec++;
         if (n_sec > 10)
         {
            Print("Error. Could not receive symbol in acceptable time.");
            SocketClose(socket);
            return false;
         }
      }

      if (!SymbolExist(symbol,isCustom))
      {
         socksend(socket, "ERROR");       // simple msg to alert python script
         Print("Error fetching symbol ",symbol);
         SocketClose(socket);
         return false;
      }
      else
      {
         socksend(socket, "SUCCESS");
      }
      // Read message with initial and final date-time separated by underscore
      inp_time_msg = socketreceive(socket,10);
      StringSplit(inp_time_msg,sep_code,msg_split);
      Print("Ticker ",symbol," from ",msg_split[0]," to ",msg_split[1]);
      d1 = StringToTime(msg_split[0]);
      d2 = StringToTime(msg_split[1]);

      // extract and send data through socket day by day
      do
      {
         to_send = "";
         // set final time of current day --
         TimeToStruct(d1,DayEnd_strc);
         DayEnd_strc.hour = 23;
         DayEnd_strc.min  = 59;
         DayEnd_strc.sec  = 00;
         DayEnd = StructToTime(DayEnd_strc);
         if (DayEnd > d2) DayEnd = d2;
         // --------------------------------

         // get intraday open-high-low-close-ticks-volume in 1-M time frame
         cp_time = CopyTime(symbol,PERIOD_M1,d1,DayEnd,time_arr);
         cp_open = CopyOpen(symbol,PERIOD_M1,d1,DayEnd,open_price);
         cp_high = CopyHigh(symbol,PERIOD_M1,d1,DayEnd,high_price);
         cp_low = CopyLow(symbol,PERIOD_M1,d1,DayEnd,low_price);
         cp_close = CopyClose(symbol,PERIOD_M1,d1,DayEnd,close_price);
         cp_tickvol = CopyTickVolume(symbol,PERIOD_M1,d1,DayEnd,tick_vol);
         cp_vol = CopyRealVolume(symbol,PERIOD_M1,d1,DayEnd,vol);

         if (cp_time > 0)
         {
            for (i = 0; i < cp_time-1; i++)
            {
               to_send += (TimeToString(time_arr[i]) + " " +
                           (string)open_price[i] + " " +
                           (string)high_price[i] + " " +
                           (string)low_price[i] + " " +
                           (string)close_price[i] + " " +
                           (string)tick_vol[i] + " " +
                           (string)vol[i] + ","
                          );
            }
            // Last set of data must not end with comma ","
            to_send += (TimeToString(time_arr[i]) + " " +
                        (string)open_price[i] + " " +
                        (string)high_price[i] + " " +
                        (string)low_price[i] + " " +
                        (string)close_price[i] + " " +
                        (string)tick_vol[i] + " " +
                        (string)vol[i]
                        );
         }
         // Move forward to the beginning of next day from last 'DayEnd'
         // For that add up 60 seconds since day-end is 23:59:00
         d1 = DayEnd + 60;
         // send data through socket connection
         socksend(socket, to_send);
         string received = socksend(socket,"_") ? socketreceive(socket, 10) : "";
      }  while (d1 < d2);
   }
   else
   {
      Print("Connection ","localhost",":",9090," error ",GetLastError());
      SocketClose(socket);
      return false;
   }

   Print("Closing Connection ... ");
   SocketClose(socket);
   Print("Done");
   return true;
}



//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
void OnStart()
{
   uint
      Nsymbols;
   int
      trials;

   Nsymbols = AssertSymbols();

   Print(Nsymbols," Valid symbols detected");

   for (uint i = 0; i < Nsymbols; i++)
   {
      trials = 0;
      // if socket conn is not opened in python or the symbol received
      // does not exist sendOHLCData will return false. Wait for 1 sec
      // among attempts up to 10 trials.
      if (!SendOHLCData()) Print("Error occurred in symbol ", i + 1);
   }

}