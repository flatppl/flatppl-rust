module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %8 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %9 = stablehlo.constant dense<"0x1872313F"> : tensor<f32>
    %10 = stablehlo.log %arg1 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %13 = stablehlo.subtract %8, %arg0 : tensor<f32>
    %14 = stablehlo.divide %13, %arg1 : tensor<f32>
    %15 = stablehlo.constant dense<-0.5> : tensor<f32>
    %16 = stablehlo.multiply %14, %14 : tensor<f32>
    %17 = stablehlo.multiply %15, %16 : tensor<f32>
    %18 = stablehlo.add %9, %11 : tensor<f32>
    %19 = stablehlo.add %18, %12 : tensor<f32>
    %20 = stablehlo.add %19, %17 : tensor<f32>
    return %20, %4 : tensor<f32>, tensor<i32>
  }
}
