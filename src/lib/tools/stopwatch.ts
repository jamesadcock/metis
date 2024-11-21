export class Stopwatch {
    private startTime: number;
    private endTime: number;
    
    public start() {
        this.startTime = new Date().getTime();
    }
    
    public stop() {
        this.endTime = new Date().getTime();
    }
    
    public getTime() {
        return this.endTime - this.startTime;
    }
}