module {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.compare GE, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.compare LE, %0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.and %4, %5 : tensor<i1>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.select %6, %0, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.constant dense<-1.1447298858494002> : tensor<f32>
    %10 = stablehlo.log %arg1 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.subtract %8, %arg0 : tensor<f32>
    %13 = stablehlo.divide %12, %arg1 : tensor<f32>
    %14 = stablehlo.multiply %13, %13 : tensor<f32>
    %15 = stablehlo.add %7, %14 : tensor<f32>
    %16 = stablehlo.log %15 : tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.add %9, %11 : tensor<f32>
    %19 = stablehlo.add %18, %17 : tensor<f32>
    %20 = stablehlo.negate %2 : tensor<f32>
    %21 = stablehlo.select %6, %19, %20 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %22 = stablehlo.subtract %2, %arg0 : tensor<f32>
    %23 = stablehlo.divide %22, %arg1 : tensor<f32>
    %24 = stablehlo.atan2 %23, %7 : tensor<f32>
    %25 = stablehlo.constant dense<0.3183098861837907> : tensor<f32>
    %26 = stablehlo.multiply %25, %24 : tensor<f32>
    %27 = stablehlo.add %0, %26 : tensor<f32>
    %28 = stablehlo.subtract %3, %arg0 : tensor<f32>
    %29 = stablehlo.divide %28, %arg1 : tensor<f32>
    %30 = stablehlo.atan2 %29, %7 : tensor<f32>
    %31 = stablehlo.multiply %25, %30 : tensor<f32>
    %32 = stablehlo.add %0, %31 : tensor<f32>
    %33 = stablehlo.subtract %27, %32 : tensor<f32>
    %34 = stablehlo.log %33 : tensor<f32>
    %35 = stablehlo.subtract %21, %34 : tensor<f32>
    return %35 : tensor<f32>
  }
}
